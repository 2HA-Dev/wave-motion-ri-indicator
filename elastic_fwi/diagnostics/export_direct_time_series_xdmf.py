from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

from dolfinx import fem, io

from elastic_fwi.diagnostics.check_sampling_conditions import (
    check_sampling_conditions,
)
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.observations.vector_array import (
    evaluate_vector_array_observation_time_series,
    vector_array_observation,
)
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    _build_materials,
    run_direct_observation_pipeline,
)
from elastic_fwi.transducers import (
    build_line_transducer_array,
    build_polyline_transducer_array,
)


def _ranked_path(path: Path, comm: MPI.Comm) -> Path:
    if comm.size == 1:
        return path

    return path.with_name(
        f"{path.stem}_rank{comm.rank:04d}of{comm.size:04d}{path.suffix}"
    )


def _build_direct_config(**kwargs: Any) -> DirectObservationPipelineConfig:
    valid_names = {
        field.name for field in dataclasses.fields(DirectObservationPipelineConfig)
    }

    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in valid_names
    }

    return DirectObservationPipelineConfig(**filtered)


def _copy_scalar_function_with_name(function: fem.Function, name: str) -> fem.Function:
    out = fem.Function(function.function_space)
    out.name = name
    out.x.array[:] = function.x.array[:]
    out.x.scatter_forward()
    return out


def _material_factor_function(materials: dict[str, fem.Function]) -> fem.Function:
    c11 = materials["C11"]
    out = fem.Function(c11.function_space)
    out.name = "material_factor"
    out.x.array[:] = c11.x.array[:] / float(LEGACY_HEALTHY_MATERIAL.C11)
    out.x.scatter_forward()
    return out


def _defect_indicator_function(materials: dict[str, fem.Function]) -> fem.Function:
    factor = _material_factor_function(materials)
    out = fem.Function(factor.function_space)
    out.name = "defect_indicator"
    out.x.array[:] = (abs(factor.x.array[:] - 1.0) > 1.0e-12).astype(float)
    out.x.scatter_forward()
    return out


def _export_static_function_vtk(
    *,
    output_path: Path,
    function: fem.Function,
    comm: MPI.Comm,
    time: float = 0.0,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with io.VTKFile(comm, str(output_path), "w") as vtk:
        vtk.write_function(function, float(time))

    return {
        "status": "ok",
        "pvd": str(output_path),
        "field_name": function.name,
        "time": float(time),
        "paraview_entrypoint": str(output_path),
    }


def _export_mesh_vtk(
    *,
    output_path: Path,
    msh,
    comm: MPI.Comm,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with io.VTKFile(comm, str(output_path), "w") as vtk:
        vtk.write_mesh(msh)

    return {
        "status": "ok",
        "pvd": str(output_path),
        "paraview_entrypoint": str(output_path),
    }


def _export_vector_time_series_vtk(
    *,
    output_path: Path,
    V,
    snapshots,
    time_loop_records,
    every_k: int,
    field_name: str,
    comm: MPI.Comm,
) -> dict[str, Any]:
    if every_k <= 0:
        raise ValueError("every_k must be positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_indices: list[int] = []
    exported_times: list[float] = []

    template = fem.Function(V)
    template.name = field_name

    with io.VTKFile(comm, str(output_path), "w") as vtk:
        for index, snapshot in enumerate(snapshots):
            if index % every_k != 0:
                continue

            record = time_loop_records[index]
            step = int(record["step"])
            time = float(record["time"])

            field = _snapshot_to_function(
                snapshot=snapshot,
                template=template,
                name=field_name,
            )

            vtk.write_function(field, time)

            exported_indices.append(step)
            exported_times.append(time)

    return {
        "status": "ok",
        "pvd": str(output_path),
        "field_name": field_name,
        "num_snapshots_available": int(len(snapshots)),
        "num_snapshots_exported": int(len(exported_indices)),
        "exported_indices": exported_indices,
        "exported_times": exported_times,
        "paraview_entrypoint": str(output_path),
    }


def _export_mesh_xdmf(
    *,
    output_dir: Path,
    msh,
    comm: MPI.Comm,
    export_xdmf: bool = True,
) -> dict[str, Any]:
    mesh_xdmf_path = output_dir / "mesh.xdmf"
    mesh_h5_path = mesh_xdmf_path.with_suffix(".h5")

    if export_xdmf:
        with io.XDMFFile(comm, str(mesh_xdmf_path), "w") as xdmf:
            xdmf.write_mesh(msh)

    vtk_export = _export_mesh_vtk(
        output_path=output_dir / "vtk" / "mesh.pvd",
        msh=msh,
        comm=comm,
    )

    return {
        "status": "ok",
        "xdmf": str(mesh_xdmf_path) if export_xdmf else None,
        "h5": str(mesh_h5_path) if export_xdmf else None,
        "vtk": vtk_export,
        "paraview_entrypoint": vtk_export["pvd"],
        "paraview_entrypoint_xdmf": str(mesh_xdmf_path) if export_xdmf else None,
    }


def _export_sampling_conditions_json(
    *,
    output_dir: Path,
    config: DirectObservationPipelineConfig,
    comm: MPI.Comm,
) -> dict[str, Any]:
    sampling_path = _ranked_path(output_dir / "sampling_conditions.json", comm)

    result = check_sampling_conditions(
        nx=config.nx,
        ny=config.ny,
        dt=config.dt,
        source_frequency=config.source_frequency,
        domain_length_x=config.domain_length_x,
        domain_length_y=config.domain_length_y,
    )

    payload = result.to_dict()

    with sampling_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)

    return {
        "status": payload["status"],
        "recommended_status": payload["recommended_status"],
        "json": str(sampling_path),
        "points_per_wavelength": payload["points_per_wavelength"],
        "time_steps_per_period": payload["time_steps_per_period"],
        "cfl": payload["cfl"],
    }


def _export_material_fields_xdmf(
    *,
    output_dir: Path,
    config: DirectObservationPipelineConfig,
    msh,
    comm: MPI.Comm,
    export_xdmf: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    material_xdmf_path = output_dir / "material_fields.xdmf"
    material_h5_path = material_xdmf_path.with_suffix(".h5")

    materials, metadata = _build_materials(msh, config)

    fields = [
        _copy_scalar_function_with_name(materials["rho"], "rho"),
        _copy_scalar_function_with_name(materials["C11"], "C11"),
        _copy_scalar_function_with_name(materials["C22"], "C22"),
        _copy_scalar_function_with_name(materials["C12"], "C12"),
        _copy_scalar_function_with_name(materials["C66"], "C66"),
        _material_factor_function(materials),
        _defect_indicator_function(materials),
    ]

    if export_xdmf:
        with io.XDMFFile(comm, str(material_xdmf_path), "w") as xdmf:
            xdmf.write_mesh(msh)
            for field in fields:
                xdmf.write_function(field, 0.0)

    vtk_exports = {}
    for field in fields:
        vtk_exports[field.name] = _export_static_function_vtk(
            output_path=output_dir / "vtk" / f"material_fields_{field.name}.pvd",
            function=field,
            comm=comm,
            time=0.0,
        )

    return {
        "status": "ok",
        "xdmf": str(material_xdmf_path) if export_xdmf else None,
        "h5": str(material_h5_path) if export_xdmf else None,
        "vtk": vtk_exports,
        "field_names": [field.name for field in fields],
        "metadata": metadata,
        "paraview_entrypoint": vtk_exports["defect_indicator"]["pvd"],
        "paraview_entrypoint_xdmf": str(material_xdmf_path) if export_xdmf else None,
    }




def _require_finite_float(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required.")

    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")

    return out


def _positive_int(value: int, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be strictly positive.")
    return out


def _resolve_receiver_sigma(
    *,
    receiver_sigma: float | None,
    receiver_array_element_length: float | None,
    source_sigma: float,
) -> float:
    if receiver_sigma is not None:
        sigma = float(receiver_sigma)
    elif receiver_array_element_length is not None:
        sigma = float(receiver_array_element_length)
    else:
        sigma = float(source_sigma)

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("receiver_sigma must be strictly positive and finite.")

    return sigma


def _prepare_receiver_geometry_for_direct_config(
    *,
    receiver_array_geometry: str,
    receiver_array_start_x: float | None,
    receiver_array_start_y: float | None,
    receiver_array_end_x: float | None,
    receiver_array_end_y: float | None,
    receiver_array_num_elements: int,
    receiver_box_xmin: float | None,
    receiver_box_xmax: float | None,
    receiver_box_ymin: float | None,
    receiver_box_ymax: float | None,
    receiver_box_top_num_elements: int,
) -> dict[str, float | int | None]:
    """Return receiver coordinates accepted by DirectObservationPipelineConfig.

    The direct pipeline validates a single line receiver when
    receiver_kind="internal_gaussian_array".  For the physical geometry
    box_except_gamma1, the actual multi-side observation is exported later from
    the same displacement snapshots.  The returned top-line coordinates are only
    a compatibility receiver; they do not trigger an additional direct solve.
    """
    receiver_array_geometry = str(receiver_array_geometry)

    if receiver_array_geometry == "line":
        return {
            "receiver_array_start_x": receiver_array_start_x,
            "receiver_array_start_y": receiver_array_start_y,
            "receiver_array_end_x": receiver_array_end_x,
            "receiver_array_end_y": receiver_array_end_y,
            "receiver_array_num_elements": int(receiver_array_num_elements),
        }

    if receiver_array_geometry != "box_except_gamma1":
        raise ValueError(
            "receiver_array_geometry must be either 'line' or 'box_except_gamma1'."
        )

    xmin = _require_finite_float(receiver_box_xmin, "receiver_box_xmin")
    xmax = _require_finite_float(receiver_box_xmax, "receiver_box_xmax")
    ymin = _require_finite_float(receiver_box_ymin, "receiver_box_ymin")
    ymax = _require_finite_float(receiver_box_ymax, "receiver_box_ymax")

    if not xmin < xmax:
        raise ValueError("receiver_box_xmin must be < receiver_box_xmax.")
    if not ymin < ymax:
        raise ValueError("receiver_box_ymin must be < receiver_box_ymax.")

    return {
        "receiver_array_start_x": (
            xmin if receiver_array_start_x is None else float(receiver_array_start_x)
        ),
        "receiver_array_start_y": (
            ymax if receiver_array_start_y is None else float(receiver_array_start_y)
        ),
        "receiver_array_end_x": (
            xmax if receiver_array_end_x is None else float(receiver_array_end_x)
        ),
        "receiver_array_end_y": (
            ymax if receiver_array_end_y is None else float(receiver_array_end_y)
        ),
        "receiver_array_num_elements": (
            _positive_int(receiver_box_top_num_elements, "receiver_box_top_num_elements")
            if int(receiver_array_num_elements) == 1
            else int(receiver_array_num_elements)
        ),
    }


def _build_receiver_vector_array_spec(
    *,
    receiver_array_geometry: str,
    receiver_array_start_x: float | None,
    receiver_array_start_y: float | None,
    receiver_array_end_x: float | None,
    receiver_array_end_y: float | None,
    receiver_array_num_elements: int,
    receiver_box_xmin: float | None,
    receiver_box_xmax: float | None,
    receiver_box_ymin: float | None,
    receiver_box_ymax: float | None,
    receiver_box_left_num_elements: int,
    receiver_box_top_num_elements: int,
    receiver_box_right_num_elements: int,
    sigma: float,
):
    receiver_array_geometry = str(receiver_array_geometry)

    if receiver_array_geometry == "line":
        return vector_array_observation(
            array=build_line_transducer_array(
                name="gamma2",
                role="transmission",
                start_x=_require_finite_float(receiver_array_start_x, "receiver_array_start_x"),
                start_y=_require_finite_float(receiver_array_start_y, "receiver_array_start_y"),
                end_x=_require_finite_float(receiver_array_end_x, "receiver_array_end_x"),
                end_y=_require_finite_float(receiver_array_end_y, "receiver_array_end_y"),
                num_elements=_positive_int(receiver_array_num_elements, "receiver_array_num_elements"),
                sigma=sigma,
                tag=2,
            ),
            name="gamma2_vector_array_transmission",
        )

    if receiver_array_geometry != "box_except_gamma1":
        raise ValueError(
            "receiver_array_geometry must be either 'line' or 'box_except_gamma1'."
        )

    xmin = _require_finite_float(receiver_box_xmin, "receiver_box_xmin")
    xmax = _require_finite_float(receiver_box_xmax, "receiver_box_xmax")
    ymin = _require_finite_float(receiver_box_ymin, "receiver_box_ymin")
    ymax = _require_finite_float(receiver_box_ymax, "receiver_box_ymax")

    if not xmin < xmax:
        raise ValueError("receiver_box_xmin must be < receiver_box_xmax.")
    if not ymin < ymax:
        raise ValueError("receiver_box_ymin must be < receiver_box_ymax.")

    return vector_array_observation(
        array=build_polyline_transducer_array(
            name="gamma2_box_except_gamma1",
            role="transmission",
            vertices=[
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
            ],
            num_elements_per_segment=[
                _positive_int(receiver_box_left_num_elements, "receiver_box_left_num_elements"),
                _positive_int(receiver_box_top_num_elements, "receiver_box_top_num_elements"),
                _positive_int(receiver_box_right_num_elements, "receiver_box_right_num_elements"),
            ],
            sigma=sigma,
            tag=2,
        ),
        name="gamma2_box_except_gamma1_vector_array_transmission",
    )


def _snapshots_with_physical_times(result) -> list[tuple[int, float, Any]]:
    records = list(result.time_loop_records)
    snapshots = list(result.displacement_snapshots)

    if len(records) != len(snapshots):
        raise RuntimeError(
            "Cannot attach physical times to displacement snapshots: "
            f"records={len(records)}, snapshots={len(snapshots)}."
        )

    return [
        (int(record["step"]), float(record["time"]), snapshot)
        for record, snapshot in zip(records, snapshots, strict=True)
    ]


def _export_receiver_vector_array_time_series(
    *,
    output_dir: Path,
    result,
    receiver_array_geometry: str,
    receiver_array_start_x: float | None,
    receiver_array_start_y: float | None,
    receiver_array_end_x: float | None,
    receiver_array_end_y: float | None,
    receiver_array_num_elements: int,
    receiver_box_xmin: float | None,
    receiver_box_xmax: float | None,
    receiver_box_ymin: float | None,
    receiver_box_ymax: float | None,
    receiver_box_left_num_elements: int,
    receiver_box_top_num_elements: int,
    receiver_box_right_num_elements: int,
    receiver_sigma: float,
    comm: MPI.Comm,
) -> dict[str, Any]:
    if result.function_space is None:
        raise RuntimeError("Direct result does not expose function_space.")

    spec = _build_receiver_vector_array_spec(
        receiver_array_geometry=receiver_array_geometry,
        receiver_array_start_x=receiver_array_start_x,
        receiver_array_start_y=receiver_array_start_y,
        receiver_array_end_x=receiver_array_end_x,
        receiver_array_end_y=receiver_array_end_y,
        receiver_array_num_elements=receiver_array_num_elements,
        receiver_box_xmin=receiver_box_xmin,
        receiver_box_xmax=receiver_box_xmax,
        receiver_box_ymin=receiver_box_ymin,
        receiver_box_ymax=receiver_box_ymax,
        receiver_box_left_num_elements=receiver_box_left_num_elements,
        receiver_box_top_num_elements=receiver_box_top_num_elements,
        receiver_box_right_num_elements=receiver_box_right_num_elements,
        sigma=receiver_sigma,
    )

    series = evaluate_vector_array_observation_time_series(
        result.function_space,
        _snapshots_with_physical_times(result),
        spec,
    )

    vector_dir = output_dir / "receiver_vector_array"
    npz_path = vector_dir / "gamma2_receiver_vector_array_time_series.npz"
    csv_path = vector_dir / "gamma2_receiver_vector_array_time_series.csv"
    manifest_path = vector_dir / "receiver_vector_array_manifest.json"

    if comm.rank == 0:
        vector_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            npz_path,
            kind=np.asarray("direct_receiver_vector_array_time_series", dtype=np.str_),
            receiver_array_geometry=np.asarray(str(receiver_array_geometry), dtype=np.str_),
            observation_name=np.asarray(series.spec.name, dtype=np.str_),
            array_name=np.asarray(series.spec.array.name, dtype=np.str_),
            role=np.asarray(series.spec.role, dtype=np.str_),
            sigma=np.asarray(series.spec.sigma, dtype=np.float64),
            num_steps=np.asarray(series.num_steps, dtype=np.int64),
            num_receivers=np.asarray(series.num_receivers, dtype=np.int64),
            steps=np.asarray(series.steps, dtype=np.int64),
            times=np.asarray(series.times, dtype=np.float64),
            centers=np.asarray(series.spec.centers, dtype=np.float64),
            values=np.asarray(series.values, dtype=np.float64),
            components=np.asarray(["x", "y"], dtype=np.str_),
        )

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "time",
                    "receiver_id",
                    "x",
                    "y",
                    "ux",
                    "uy",
                ],
            )
            writer.writeheader()
            centers = np.asarray(series.spec.centers, dtype=float)
            for k in range(series.num_steps):
                for j in range(series.num_receivers):
                    writer.writerow(
                        {
                            "step": int(series.steps[k]),
                            "time": float(series.times[k]),
                            "receiver_id": int(j),
                            "x": float(centers[j, 0]),
                            "y": float(centers[j, 1]),
                            "ux": float(series.values[k, j, 0]),
                            "uy": float(series.values[k, j, 1]),
                        }
                    )

        manifest = {
            "kind": "direct_receiver_vector_array_manifest",
            "status": "ok",
            "receiver_array_geometry": str(receiver_array_geometry),
            "receiver_sigma": float(receiver_sigma),
            "num_steps": int(series.num_steps),
            "num_receivers": int(series.num_receivers),
            "components": ["x", "y"],
            "spec": series.spec.as_dict(),
            "files": {
                "npz": str(npz_path),
                "csv": str(csv_path),
            },
        }

        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    return {
        "status": "ok",
        "receiver_array_geometry": str(receiver_array_geometry),
        "receiver_sigma": float(receiver_sigma),
        "num_steps": int(series.num_steps),
        "num_receivers": int(series.num_receivers),
        "npz": str(npz_path),
        "csv": str(csv_path),
        "manifest": str(manifest_path),
        "spec": series.spec.as_dict(),
    }

def _snapshot_to_function(
    *,
    snapshot: Any,
    template: fem.Function,
    name: str,
) -> fem.Function:
    out = fem.Function(template.function_space)
    out.name = name

    if isinstance(snapshot, fem.Function):
        out.x.array[:] = snapshot.x.array[:]
        out.x.scatter_forward()
        return out

    values = np.asarray(snapshot, dtype=float).reshape(-1)

    if values.size > out.x.array.size:
        raise ValueError(
            "Snapshot size is larger than local function vector size: "
            f"snapshot={values.size}, function={out.x.array.size}."
        )

    # In MPI, stored snapshots may contain only owned local dofs, while
    # fem.Function.x.array contains owned dofs followed by ghost entries.
    # Fill the owned prefix, then update ghost entries with scatter_forward.
    out.x.array[:] = 0.0
    out.x.array[: values.size] = values
    out.x.scatter_forward()

    return out


def export_direct_time_series_xdmf(
    *,
    output_dir: Path,
    nx: int,
    ny: int,
    degree: int = 1,
    mesh_kind: str = "structured",
    mesh_far_h: float | None = None,
    mesh_near_h: float | None = None,
    mesh_refine_source: bool = False,
    mesh_refine_defect: bool = False,
    mesh_refine_from_defects_json: bool = False,
    mesh_refine_source_radius: float | None = None,
    mesh_refine_defect_radius: float | None = None,
    mesh_auto_from_wavelength: bool = False,
    mesh_target_ppw: float = 10.0,
    mesh_near_target_ppw: float = 20.0,
    mesh_near_stiffness_factor: float = 1.0,
    absorbing_layer: bool = False,
    absorbing_layer_width: float | None = None,
    absorbing_layer_width_wavelengths: float = 1.5,
    absorbing_layer_min_cells: int = 6,
    absorbing_layer_strength: float = 3.0,
    absorbing_layer_polynomial_order: float = 2.0,
    absorbing_layer_max_fraction: float = 0.45,
    domain_length_x: float = 1.0,
    domain_length_y: float = 1.0,
    source_kind: str = "boundary_segment",
    source_center_x: float = 0.5,
    source_center_y: float = 0.5,
    source_radius: float = 0.05,
    source_sigma: float = 0.05,
    source_direction_x: float = 0.0,
    source_direction_y: float = 1.0,
    source_side: str = "bottom",
    source_center: float | None = None,
    source_length: float | None = None,
    source_array_start: float | None = None,
    source_array_end: float | None = None,
    source_array_num_elements: int = 1,
    source_array_element_length: float | None = None,
    source_array_start_x: float | None = None,
    source_array_start_y: float | None = None,
    source_array_end_x: float | None = None,
    source_array_end_y: float | None = None,
    receiver_kind: str = "boundary_segment",
    receiver_side: str = "top",
    receiver_center: float | None = None,
    receiver_length: float | None = None,
    receiver_array_start: float | None = None,
    receiver_array_end: float | None = None,
    receiver_array_num_elements: int = 1,
    receiver_array_element_length: float | None = None,
    receiver_array_start_x: float | None = None,
    receiver_array_start_y: float | None = None,
    receiver_array_end_x: float | None = None,
    receiver_array_end_y: float | None = None,
    receiver_sigma: float | None = None,
    receiver_array_geometry: str = "line",
    receiver_box_xmin: float | None = None,
    receiver_box_xmax: float | None = None,
    receiver_box_ymin: float | None = None,
    receiver_box_ymax: float | None = None,
    receiver_box_left_num_elements: int = 13,
    receiver_box_top_num_elements: int = 11,
    receiver_box_right_num_elements: int = 13,
    export_receiver_vector_array: bool = True,
    auto_place_probes_inside_absorbing_layer: bool = False,
    source_segment_xmin: float = 0.0,
    source_segment_xmax: float = 1.0,
    receiver_segment_xmin: float = 0.0,
    receiver_segment_xmax: float = 1.0,
    material_kind: str = "healthy",
    defect_center_x: float = 0.5,
    defect_center_y: float = 0.5,
    defect_radius: float = 0.35,
    contrast_factor: float = 0.8,
    defects_json: str | None = None,
    defect_background_factor: float = 1.0,
    defect_conflict_policy: str = "last",
    theta: float = 0.25,
    dt: float = 0.1,
    num_steps: int = 5,
    source_frequency: float = 1.0,
    every_k: int = 1,
    visual_output_format: str = "pvd",
    export_material_fields: bool = True,
    export_mesh: bool = True,
    export_sampling_conditions: bool = True,
    reuse_effective_solver: bool = True,
    ksp_type: str = "cg",
    pc_type: str = "jacobi",
    ksp_rtol: float = 1.0e-10,
    ksp_atol: float = 1.0e-12,
    ksp_max_it: int = 1000,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    if every_k <= 0:
        raise ValueError("every_k must be positive.")

    visual_output_format = str(visual_output_format).lower()
    if visual_output_format not in {"pvd", "xdmf", "both"}:
        raise ValueError(
            "visual_output_format must be one of 'pvd', 'xdmf', or 'both'."
        )

    export_pvd = visual_output_format in {"pvd", "both"}
    export_xdmf = visual_output_format in {"xdmf", "both"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # XDMF/HDF5 is a collective parallel output. All MPI ranks must open the
    # same file path with the same communicator. Only JSON manifests are ranked.
    xdmf_path = output_dir / "direct_displacement.xdmf"
    manifest_path = _ranked_path(output_dir / "direct_time_series_manifest.json", comm)

    receiver_array_geometry = str(receiver_array_geometry)
    receiver_geometry_for_config = _prepare_receiver_geometry_for_direct_config(
        receiver_array_geometry=receiver_array_geometry,
        receiver_array_start_x=receiver_array_start_x,
        receiver_array_start_y=receiver_array_start_y,
        receiver_array_end_x=receiver_array_end_x,
        receiver_array_end_y=receiver_array_end_y,
        receiver_array_num_elements=receiver_array_num_elements,
        receiver_box_xmin=receiver_box_xmin,
        receiver_box_xmax=receiver_box_xmax,
        receiver_box_ymin=receiver_box_ymin,
        receiver_box_ymax=receiver_box_ymax,
        receiver_box_top_num_elements=receiver_box_top_num_elements,
    )
    effective_receiver_array_start_x = receiver_geometry_for_config["receiver_array_start_x"]
    effective_receiver_array_start_y = receiver_geometry_for_config["receiver_array_start_y"]
    effective_receiver_array_end_x = receiver_geometry_for_config["receiver_array_end_x"]
    effective_receiver_array_end_y = receiver_geometry_for_config["receiver_array_end_y"]
    effective_receiver_array_num_elements = int(
        receiver_geometry_for_config["receiver_array_num_elements"]
    )

    config = _build_direct_config(
        nx=nx,
        ny=ny,
        degree=degree,
        mesh_kind=mesh_kind,
        mesh_far_h=mesh_far_h,
        mesh_near_h=mesh_near_h,
        mesh_refine_source=mesh_refine_source,
        mesh_refine_defect=mesh_refine_defect,
        mesh_refine_from_defects_json=mesh_refine_from_defects_json,
        mesh_refine_source_radius=mesh_refine_source_radius,
        mesh_refine_defect_radius=mesh_refine_defect_radius,
        mesh_auto_from_wavelength=mesh_auto_from_wavelength,
        mesh_target_ppw=mesh_target_ppw,
        mesh_near_target_ppw=mesh_near_target_ppw,
        mesh_near_stiffness_factor=mesh_near_stiffness_factor,
        absorbing_layer=absorbing_layer,
        absorbing_layer_width=absorbing_layer_width,
        absorbing_layer_width_wavelengths=absorbing_layer_width_wavelengths,
        absorbing_layer_min_cells=absorbing_layer_min_cells,
        absorbing_layer_strength=absorbing_layer_strength,
        absorbing_layer_polynomial_order=absorbing_layer_polynomial_order,
        absorbing_layer_max_fraction=absorbing_layer_max_fraction,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        source_kind=source_kind,
        source_center_x=source_center_x,
        source_center_y=source_center_y,
        source_radius=source_radius,
        source_sigma=source_sigma,
        source_direction_x=source_direction_x,
        source_direction_y=source_direction_y,
        source_side=source_side,
        source_center=source_center,
        source_length=source_length,
        source_array_start=source_array_start,
        source_array_end=source_array_end,
        source_array_num_elements=source_array_num_elements,
        source_array_element_length=source_array_element_length,
        source_array_start_x=source_array_start_x,
        source_array_start_y=source_array_start_y,
        source_array_end_x=source_array_end_x,
        source_array_end_y=source_array_end_y,
        receiver_kind=receiver_kind,
        receiver_side=receiver_side,
        receiver_center=receiver_center,
        receiver_length=receiver_length,
        receiver_array_start=receiver_array_start,
        receiver_array_end=receiver_array_end,
        receiver_array_num_elements=effective_receiver_array_num_elements,
        receiver_array_element_length=receiver_array_element_length,
        receiver_array_start_x=effective_receiver_array_start_x,
        receiver_array_start_y=effective_receiver_array_start_y,
        receiver_array_end_x=effective_receiver_array_end_x,
        receiver_array_end_y=effective_receiver_array_end_y,
        auto_place_probes_inside_absorbing_layer=auto_place_probes_inside_absorbing_layer,
        source_segment_xmin=source_segment_xmin,
        source_segment_xmax=source_segment_xmax,
        receiver_segment_xmin=receiver_segment_xmin,
        receiver_segment_xmax=receiver_segment_xmax,
        material_kind=material_kind,
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
        contrast_factor=contrast_factor,
        defects_json=defects_json,
        defect_background_factor=defect_background_factor,
        defect_conflict_policy=defect_conflict_policy,
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        source_frequency=source_frequency,
        store_displacement_snapshots=True,
        reuse_effective_solver=reuse_effective_solver,
        ksp_type=ksp_type,
        pc_type=pc_type,
        ksp_rtol=ksp_rtol,
        ksp_atol=ksp_atol,
        ksp_max_it=ksp_max_it,
    )

    result = run_direct_observation_pipeline(config=config, comm=comm)

    snapshots = list(result.displacement_snapshots)

    if not snapshots:
        raise RuntimeError("No direct displacement snapshots were produced.")

    if not hasattr(result, "function_space"):
        raise RuntimeError(
            "DirectObservationPipelineResult does not expose function_space."
        )

    V = result.function_space

    u_template = fem.Function(V)
    u_template.name = "direct_displacement"

    direct_vtk_export: dict[str, Any] | None = None
    direct_xdmf_export: dict[str, Any] | None = None

    if export_pvd:
        direct_vtk_export = _export_vector_time_series_vtk(
            output_path=output_dir / "vtk" / "direct_displacement.pvd",
            V=V,
            snapshots=snapshots,
            time_loop_records=result.time_loop_records,
            every_k=every_k,
            field_name="direct_displacement",
            comm=comm,
        )

    exported_indices: list[int] = []
    exported_times: list[float] = []

    if export_xdmf:
        with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
            xdmf.write_mesh(V.mesh)

            for index, snapshot in enumerate(snapshots):
                if index % every_k != 0:
                    continue

                record = result.time_loop_records[index]
                step = int(record["step"])
                time = float(record["time"])

                u = _snapshot_to_function(
                    snapshot=snapshot,
                    template=u_template,
                    name="direct_displacement",
                )

                xdmf.write_function(u, time)

                exported_indices.append(step)
                exported_times.append(time)

        direct_xdmf_export = {
            "status": "ok",
            "xdmf": str(xdmf_path),
            "h5": str(xdmf_path.with_suffix(".h5")),
            "paraview_entrypoint": str(xdmf_path),
        }
    elif direct_vtk_export is not None:
        exported_indices = list(direct_vtk_export["exported_indices"])
        exported_times = list(direct_vtk_export["exported_times"])

    h5_path = xdmf_path.with_suffix(".h5") if export_xdmf else None
    paraview_entrypoint = (
        direct_vtk_export["pvd"] if direct_vtk_export is not None else str(xdmf_path)
    )

    material_export: dict[str, Any] | None = None
    if export_material_fields:
        material_export = _export_material_fields_xdmf(
            output_dir=output_dir,
            config=config,
            msh=V.mesh,
            comm=comm,
            export_xdmf=export_xdmf,
        )

    mesh_export: dict[str, Any] | None = None
    if export_mesh:
        mesh_export = _export_mesh_xdmf(
            output_dir=output_dir,
            msh=V.mesh,
            comm=comm,
            export_xdmf=export_xdmf,
        )

    sampling_export: dict[str, Any] | None = None
    if export_sampling_conditions:
        sampling_export = _export_sampling_conditions_json(
            output_dir=output_dir,
            config=config,
            comm=comm,
        )

    receiver_vector_array_export: dict[str, Any] | None = None
    should_export_receiver_vector_array = bool(export_receiver_vector_array) and (
        receiver_kind == "internal_gaussian_array"
        or receiver_array_geometry == "box_except_gamma1"
    )

    if should_export_receiver_vector_array:
        resolved_receiver_sigma = _resolve_receiver_sigma(
            receiver_sigma=receiver_sigma,
            receiver_array_element_length=receiver_array_element_length,
            source_sigma=source_sigma,
        )
        receiver_vector_array_export = _export_receiver_vector_array_time_series(
            output_dir=output_dir,
            result=result,
            receiver_array_geometry=receiver_array_geometry,
            receiver_array_start_x=effective_receiver_array_start_x,
            receiver_array_start_y=effective_receiver_array_start_y,
            receiver_array_end_x=effective_receiver_array_end_x,
            receiver_array_end_y=effective_receiver_array_end_y,
            receiver_array_num_elements=effective_receiver_array_num_elements,
            receiver_box_xmin=receiver_box_xmin,
            receiver_box_xmax=receiver_box_xmax,
            receiver_box_ymin=receiver_box_ymin,
            receiver_box_ymax=receiver_box_ymax,
            receiver_box_left_num_elements=receiver_box_left_num_elements,
            receiver_box_top_num_elements=receiver_box_top_num_elements,
            receiver_box_right_num_elements=receiver_box_right_num_elements,
            receiver_sigma=resolved_receiver_sigma,
            comm=comm,
        )

    manifest = {
        "kind": f"direct_time_series_{visual_output_format}",
        "status": "ok",
        "mpi_rank": comm.rank,
        "mpi_size": comm.size,
        "nx": int(nx),
        "ny": int(ny),
        "degree": int(degree),
        "domain_length_x": float(domain_length_x),
        "domain_length_y": float(domain_length_y),
        "source_kind": str(source_kind),
        "source_center_x": float(source_center_x),
        "source_center_y": float(source_center_y),
        "source_radius": float(source_radius),
        "source_sigma": float(source_sigma),
        "source_direction_x": float(source_direction_x),
        "source_direction_y": float(source_direction_y),
        "source_side": str(source_side),
        "source_center": None if source_center is None else float(source_center),
        "source_length": None if source_length is None else float(source_length),
        "source_array_start": None if source_array_start is None else float(source_array_start),
        "source_array_end": None if source_array_end is None else float(source_array_end),
        "source_array_num_elements": int(source_array_num_elements),
        "source_array_element_length": None if source_array_element_length is None else float(source_array_element_length),
        "source_array_start_x": None if source_array_start_x is None else float(source_array_start_x),
        "source_array_start_y": None if source_array_start_y is None else float(source_array_start_y),
        "source_array_end_x": None if source_array_end_x is None else float(source_array_end_x),
        "source_array_end_y": None if source_array_end_y is None else float(source_array_end_y),
        "receiver_kind": str(receiver_kind),
        "receiver_side": str(receiver_side),
        "receiver_center": None if receiver_center is None else float(receiver_center),
        "receiver_length": None if receiver_length is None else float(receiver_length),
        "receiver_array_start": None if receiver_array_start is None else float(receiver_array_start),
        "receiver_array_end": None if receiver_array_end is None else float(receiver_array_end),
        "receiver_array_num_elements": int(receiver_array_num_elements),
        "receiver_array_element_length": None if receiver_array_element_length is None else float(receiver_array_element_length),
        "receiver_array_start_x": None if receiver_array_start_x is None else float(receiver_array_start_x),
        "receiver_array_start_y": None if receiver_array_start_y is None else float(receiver_array_start_y),
        "receiver_array_end_x": None if receiver_array_end_x is None else float(receiver_array_end_x),
        "receiver_array_end_y": None if receiver_array_end_y is None else float(receiver_array_end_y),
        "effective_receiver_array_start_x": None if effective_receiver_array_start_x is None else float(effective_receiver_array_start_x),
        "effective_receiver_array_start_y": None if effective_receiver_array_start_y is None else float(effective_receiver_array_start_y),
        "effective_receiver_array_end_x": None if effective_receiver_array_end_x is None else float(effective_receiver_array_end_x),
        "effective_receiver_array_end_y": None if effective_receiver_array_end_y is None else float(effective_receiver_array_end_y),
        "effective_receiver_array_num_elements": int(effective_receiver_array_num_elements),
        "receiver_sigma": None if receiver_sigma is None else float(receiver_sigma),
        "receiver_array_geometry": str(receiver_array_geometry),
        "receiver_box_xmin": None if receiver_box_xmin is None else float(receiver_box_xmin),
        "receiver_box_xmax": None if receiver_box_xmax is None else float(receiver_box_xmax),
        "receiver_box_ymin": None if receiver_box_ymin is None else float(receiver_box_ymin),
        "receiver_box_ymax": None if receiver_box_ymax is None else float(receiver_box_ymax),
        "receiver_box_left_num_elements": int(receiver_box_left_num_elements),
        "receiver_box_top_num_elements": int(receiver_box_top_num_elements),
        "receiver_box_right_num_elements": int(receiver_box_right_num_elements),
        "export_receiver_vector_array": bool(export_receiver_vector_array),
        "receiver_vector_array": receiver_vector_array_export,
        "auto_place_probes_inside_absorbing_layer": bool(auto_place_probes_inside_absorbing_layer),
        "source_segment_xmin": float(source_segment_xmin),
        "source_segment_xmax": float(source_segment_xmax),
        "receiver_segment_xmin": float(receiver_segment_xmin),
        "receiver_segment_xmax": float(receiver_segment_xmax),
        "material_kind": str(material_kind),
        "defect_center_x": float(defect_center_x),
        "defect_center_y": float(defect_center_y),
        "defect_radius": float(defect_radius),
        "contrast_factor": float(contrast_factor),
        "defects_json": None if defects_json is None else str(defects_json),
        "defect_background_factor": float(defect_background_factor),
        "defect_conflict_policy": str(defect_conflict_policy),
        "theta": float(theta),
        "dt": float(dt),
        "num_steps": int(num_steps),
        "source_frequency": float(source_frequency),
        "every_k": int(every_k),
        "visual_output_format": visual_output_format,
        "export_pvd": bool(export_pvd),
        "export_xdmf": bool(export_xdmf),
        "reuse_effective_solver": bool(reuse_effective_solver),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "ksp_rtol": float(ksp_rtol),
        "ksp_atol": float(ksp_atol),
        "ksp_max_it": int(ksp_max_it),
        "num_snapshots_available": len(snapshots),
        "num_snapshots_exported": len(exported_indices),
        "exported_indices": exported_indices,
        "exported_times": exported_times,
        "field_name": "direct_displacement",
        "mesh_metadata": result.mesh_metadata,
        "absorbing_layer": result.mesh_metadata.get("absorbing_layer", {"enabled": False}),
        "mesh_kind": result.mesh_metadata.get("mesh_kind"),
        "mesh_far_h": result.mesh_metadata.get("mesh_far_h"),
        "mesh_near_h": result.mesh_metadata.get("mesh_near_h"),
        "mesh_refine_source": result.mesh_metadata.get("mesh_refine_source"),
        "mesh_refine_defect": result.mesh_metadata.get("mesh_refine_defect"),
        "mesh_refine_source_radius": result.mesh_metadata.get("mesh_refine_source_radius"),
        "mesh_refine_defect_radius": result.mesh_metadata.get("mesh_refine_defect_radius"),
        "mesh_auto_from_wavelength": result.mesh_metadata.get("mesh_auto_from_wavelength"),
        "mesh_target_ppw": result.mesh_metadata.get("mesh_target_ppw"),
        "mesh_near_target_ppw": result.mesh_metadata.get("mesh_near_target_ppw"),
        "wavelength_mesh_sizing": result.mesh_metadata.get("wavelength_mesh_sizing"),
        "effective_nx": result.mesh_metadata.get("effective_nx"),
        "effective_ny": result.mesh_metadata.get("effective_ny"),
        "refinement_disks": result.mesh_metadata.get("refinement_disks", []),
        "source_metadata": result.source_metadata,
        "observation_metadata": result.observation_metadata,
        "export_material_fields": bool(export_material_fields),
        "material_fields": material_export,
        "material_xdmf": None if material_export is None else material_export["xdmf"],
        "material_h5": None if material_export is None else material_export["h5"],
        "export_mesh": bool(export_mesh),
        "mesh": mesh_export,
        "mesh_xdmf": None if mesh_export is None else mesh_export["xdmf"],
        "mesh_h5": None if mesh_export is None else mesh_export["h5"],
        "export_sampling_conditions": bool(export_sampling_conditions),
        "sampling_conditions": sampling_export,
        "sampling_json": None if sampling_export is None else sampling_export["json"],
        "xdmf": str(xdmf_path) if export_xdmf else None,
        "h5": str(h5_path) if h5_path is not None else None,
        "pvd": None if direct_vtk_export is None else direct_vtk_export["pvd"],
        "direct_displacement": {
            "vtk": direct_vtk_export,
            "xdmf": direct_xdmf_export,
        },
        "paraview_entrypoint": paraview_entrypoint,
    }

    if comm.rank == 0:
        try:
            from elastic_fwi.tracking.tracked_manifest import enrich_manifest_with_tracking

            manifest = enrich_manifest_with_tracking(
                manifest,
                output_dir=output_dir,
                args={
                    "diagnostic": "export_direct_time_series_xdmf",
                    "nx": int(nx),
                    "ny": int(ny),
                    "degree": int(degree),
                    "dt": float(dt),
                    "num_steps": int(num_steps),
                    "every_k": int(every_k),
                    "visual_output_format": visual_output_format,
                    "export_material_fields": bool(export_material_fields),
                    "export_mesh": bool(export_mesh),
                    "export_sampling_conditions": bool(export_sampling_conditions),
                },
                comm=comm,
            )
        except Exception as exc:
            manifest["tracking_error"] = repr(exc)
    else:
        manifest["tracking"] = {
            "schema_version": "elastic_fwi_tracking_v1",
            "status": "skipped_non_root_rank",
            "reason": "tracking sidecars are written by MPI rank 0 only",
            "mpi_rank": int(comm.rank),
            "mpi_size": int(comm.size),
        }

    if comm.rank == 0:
        try:
            from elastic_fwi.tracking.tracked_manifest import enrich_manifest_with_tracking

            manifest = enrich_manifest_with_tracking(
                manifest,
                output_dir=output_dir,
                args={
                    "diagnostic": "export_direct_time_series_xdmf",
                    "nx": int(nx),
                    "ny": int(ny),
                    "degree": int(degree),
                    "dt": float(dt),
                    "num_steps": int(num_steps),
                    "every_k": int(every_k),
                    "visual_output_format": visual_output_format,
                    "export_material_fields": bool(export_material_fields),
                    "export_mesh": bool(export_mesh),
                    "export_sampling_conditions": bool(export_sampling_conditions),
                },
                comm=comm,
            )
        except Exception as exc:
            manifest["tracking_error"] = repr(exc)
    else:
        manifest["tracking"] = {
            "schema_version": "elastic_fwi_tracking_v1",
            "status": "skipped_non_root_rank",
            "reason": "tracking sidecars are written by MPI rank 0 only",
            "mpi_rank": int(comm.rank),
            "mpi_size": int(comm.size),
        }

    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)

    if comm.rank == 0:
        print(
            f"[DIRECT TIME SERIES {visual_output_format.upper()}] OK "
            f"output={paraview_entrypoint}, snapshots={len(exported_indices)}, "
            f"MPI size={comm.size}"
        )

    return {
        "status": "ok",
        "xdmf": str(xdmf_path) if export_xdmf else None,
        "h5": str(h5_path) if h5_path is not None else None,
        "pvd": None if direct_vtk_export is None else direct_vtk_export["pvd"],
        "manifest": str(manifest_path),
        "material_xdmf": None if material_export is None else material_export["xdmf"],
        "material_h5": None if material_export is None else material_export["h5"],
        "mesh_xdmf": None if mesh_export is None else mesh_export["xdmf"],
        "mesh_h5": None if mesh_export is None else mesh_export["h5"],
        "sampling_json": None if sampling_export is None else sampling_export["json"],
        "receiver_vector_array_manifest": (
            None if receiver_vector_array_export is None else receiver_vector_array_export["manifest"]
        ),
        "receiver_vector_array_npz": (
            None if receiver_vector_array_export is None else receiver_vector_array_export["npz"]
        ),
        "receiver_vector_array_csv": (
            None if receiver_vector_array_export is None else receiver_vector_array_export["csv"]
        ),
        "num_snapshots_exported": len(exported_indices),
        "paraview_entrypoint": paraview_entrypoint,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export direct displacement time series to XDMF/HDF5 for ParaView."
    )

    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--mesh-kind", type=str, default="structured", choices=("structured", "gmsh_refined"))
    parser.add_argument("--mesh-far-h", type=float, default=None)
    parser.add_argument("--mesh-near-h", type=float, default=None)
    parser.add_argument("--mesh-refine-source", action="store_true")
    parser.add_argument("--mesh-refine-defect", action="store_true")
    parser.add_argument("--mesh-refine-from-defects-json", action="store_true")
    parser.add_argument("--mesh-refine-source-radius", type=float, default=None)
    parser.add_argument("--mesh-refine-defect-radius", type=float, default=None)
    parser.add_argument("--mesh-auto-from-wavelength", action="store_true")
    parser.add_argument("--mesh-target-ppw", type=float, default=10.0)
    parser.add_argument("--mesh-near-target-ppw", type=float, default=20.0)
    parser.add_argument("--mesh-near-stiffness-factor", type=float, default=1.0)
    parser.add_argument("--absorbing-layer", action="store_true")
    parser.add_argument("--absorbing-layer-width", type=float, default=None)
    parser.add_argument("--absorbing-layer-width-wavelengths", type=float, default=1.5)
    parser.add_argument("--absorbing-layer-min-cells", type=int, default=6)
    parser.add_argument("--absorbing-layer-strength", type=float, default=3.0)
    parser.add_argument("--absorbing-layer-polynomial-order", type=float, default=2.0)
    parser.add_argument("--absorbing-layer-max-fraction", type=float, default=0.45)
    parser.add_argument("--domain-length-x", type=float, default=1.0)
    parser.add_argument("--domain-length-y", type=float, default=1.0)
    parser.add_argument("--source-kind", type=str, default="boundary_segment", choices=("boundary_segment", "boundary_point_array", "internal_disk", "internal_gaussian", "internal_gaussian_array"))
    parser.add_argument("--source-center-x", type=float, default=0.5)
    parser.add_argument("--source-center-y", type=float, default=0.5)
    parser.add_argument("--source-radius", type=float, default=0.05)
    parser.add_argument("--source-sigma", type=float, default=0.05)
    parser.add_argument("--source-direction-x", type=float, default=0.0)
    parser.add_argument("--source-direction-y", type=float, default=1.0)
    parser.add_argument("--source-side", type=str, default="bottom", choices=("bottom", "top", "left", "right"))
    parser.add_argument("--source-center", type=float, default=None)
    parser.add_argument("--source-length", type=float, default=None)
    parser.add_argument("--source-array-start", type=float, default=None)
    parser.add_argument("--source-array-end", type=float, default=None)
    parser.add_argument("--source-array-num-elements", type=int, default=1)
    parser.add_argument("--source-array-element-length", type=float, default=None)
    parser.add_argument("--source-array-start-x", type=float, default=None)
    parser.add_argument("--source-array-start-y", type=float, default=None)
    parser.add_argument("--source-array-end-x", type=float, default=None)
    parser.add_argument("--source-array-end-y", type=float, default=None)
    parser.add_argument("--receiver-kind", type=str, default="boundary_segment", choices=("boundary_segment", "boundary_point_array", "internal_disk", "internal_gaussian", "internal_gaussian_array"))
    parser.add_argument("--receiver-side", type=str, default="top", choices=("bottom", "top", "left", "right"))
    parser.add_argument("--receiver-center", type=float, default=None)
    parser.add_argument("--receiver-length", type=float, default=None)
    parser.add_argument("--receiver-array-start", type=float, default=None)
    parser.add_argument("--receiver-array-end", type=float, default=None)
    parser.add_argument("--receiver-array-num-elements", type=int, default=1)
    parser.add_argument("--receiver-array-element-length", type=float, default=None)
    parser.add_argument("--receiver-array-start-x", type=float, default=None)
    parser.add_argument("--receiver-array-start-y", type=float, default=None)
    parser.add_argument("--receiver-array-end-x", type=float, default=None)
    parser.add_argument("--receiver-array-end-y", type=float, default=None)
    parser.add_argument(
        "--receiver-sigma",
        type=float,
        default=None,
        help=(
            "Gaussian width for exported vector-array receiver observations. "
            "If omitted, receiver-array-element-length is used, then source-sigma."
        ),
    )
    parser.add_argument(
        "--receiver-array-geometry",
        type=str,
        default="line",
        choices=("line", "box_except_gamma1"),
        help=(
            "Geometry used for exported internal vector-array receivers. "
            "box_except_gamma1 exports left+top+right sides from one direct solve."
        ),
    )
    parser.add_argument("--receiver-box-xmin", type=float, default=None)
    parser.add_argument("--receiver-box-xmax", type=float, default=None)
    parser.add_argument("--receiver-box-ymin", type=float, default=None)
    parser.add_argument("--receiver-box-ymax", type=float, default=None)
    parser.add_argument("--receiver-box-left-num-elements", type=int, default=13)
    parser.add_argument("--receiver-box-top-num-elements", type=int, default=11)
    parser.add_argument("--receiver-box-right-num-elements", type=int, default=13)
    receiver_vector_export_group = parser.add_mutually_exclusive_group()
    receiver_vector_export_group.add_argument(
        "--export-receiver-vector-array",
        dest="export_receiver_vector_array",
        action="store_true",
        default=True,
        help="Export receiver vector-array time series after the single direct solve.",
    )
    receiver_vector_export_group.add_argument(
        "--no-export-receiver-vector-array",
        dest="export_receiver_vector_array",
        action="store_false",
        help="Disable receiver vector-array time-series export.",
    )
    parser.add_argument(
        "--auto-place-probes-inside-absorbing-layer",
        action="store_true",
        help=(
            "Automatically place internal source/receiver arrays on the inner "
            "edge of the absorbing layer: source at y=width and receiver at "
            "y=domain_length_y-width."
        ),
    )
    parser.add_argument("--source-segment-xmin", type=float, default=0.0)
    parser.add_argument("--source-segment-xmax", type=float, default=1.0)
    parser.add_argument("--receiver-segment-xmin", type=float, default=0.0)
    parser.add_argument("--receiver-segment-xmax", type=float, default=1.0)
    parser.add_argument("--material-kind", type=str, default="healthy")
    parser.add_argument("--defect-center-x", type=float, default=0.5)
    parser.add_argument("--defect-center-y", type=float, default=0.5)
    parser.add_argument("--defect-radius", type=float, default=0.35)
    parser.add_argument("--contrast-factor", type=float, default=0.8)
    parser.add_argument("--defects-json", type=str, default=None)
    parser.add_argument("--defect-background-factor", type=float, default=1.0)
    parser.add_argument(
        "--defect-conflict-policy",
        type=str,
        default="last",
        choices=("first", "last", "error"),
    )
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--source-frequency", type=float, default=1.0)
    parser.add_argument("--every-k", type=int, default=1)
    parser.add_argument(
        "--visual-output-format",
        type=str,
        default="pvd",
        choices=("pvd", "xdmf", "both"),
        help=(
            "ParaView visual output format. Default 'pvd' writes VTK/PVD files only "
            "for visualizable fields; 'xdmf' restores the historical XDMF/HDF5 "
            "output; 'both' writes both families."
        ),
    )

    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse-effective-solver",
        dest="reuse_effective_solver",
        action="store_true",
        default=True,
        help="Reuse the fixed Newmark effective KSP/PC across all time steps.",
    )
    reuse_group.add_argument(
        "--no-reuse-effective-solver",
        dest="reuse_effective_solver",
        action="store_false",
        help="Rebuild the effective KSP/PC at each time step.",
    )
    parser.add_argument("--ksp-type", type=str, default="cg")
    parser.add_argument("--pc-type", type=str, default="jacobi")
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--ksp-atol", type=float, default=1.0e-12)
    parser.add_argument("--ksp-max-it", type=int, default=1000)

    parser.add_argument(
        "--no-export-material-fields",
        action="store_true",
        help="Disable XDMF/HDF5 export of static DG0 material fields.",
    )
    parser.add_argument(
        "--no-export-mesh",
        action="store_true",
        help="Disable XDMF/HDF5 export of the computational mesh.",
    )
    parser.add_argument(
        "--no-export-sampling-conditions",
        action="store_true",
        help="Disable JSON export of Shannon/CFL sampling diagnostics.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    export_direct_time_series_xdmf(
        output_dir=args.output_dir,
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        mesh_kind=args.mesh_kind,
        mesh_far_h=args.mesh_far_h,
        mesh_near_h=args.mesh_near_h,
        mesh_refine_source=args.mesh_refine_source,
        mesh_refine_defect=args.mesh_refine_defect,
        mesh_refine_from_defects_json=args.mesh_refine_from_defects_json,
        mesh_refine_source_radius=args.mesh_refine_source_radius,
        mesh_refine_defect_radius=args.mesh_refine_defect_radius,
        mesh_auto_from_wavelength=args.mesh_auto_from_wavelength,
        mesh_target_ppw=args.mesh_target_ppw,
        mesh_near_target_ppw=args.mesh_near_target_ppw,
        mesh_near_stiffness_factor=args.mesh_near_stiffness_factor,
        absorbing_layer=args.absorbing_layer,
        absorbing_layer_width=args.absorbing_layer_width,
        absorbing_layer_width_wavelengths=args.absorbing_layer_width_wavelengths,
        absorbing_layer_min_cells=args.absorbing_layer_min_cells,
        absorbing_layer_strength=args.absorbing_layer_strength,
        absorbing_layer_polynomial_order=args.absorbing_layer_polynomial_order,
        absorbing_layer_max_fraction=args.absorbing_layer_max_fraction,
        domain_length_x=args.domain_length_x,
        domain_length_y=args.domain_length_y,
        source_kind=args.source_kind,
        source_center_x=args.source_center_x,
        source_center_y=args.source_center_y,
        source_radius=args.source_radius,
        source_sigma=args.source_sigma,
        source_direction_x=args.source_direction_x,
        source_direction_y=args.source_direction_y,
        source_side=args.source_side,
        source_center=args.source_center,
        source_length=args.source_length,
        source_array_start=args.source_array_start,
        source_array_end=args.source_array_end,
        source_array_num_elements=args.source_array_num_elements,
        source_array_element_length=args.source_array_element_length,
        source_array_start_x=args.source_array_start_x,
        source_array_start_y=args.source_array_start_y,
        source_array_end_x=args.source_array_end_x,
        source_array_end_y=args.source_array_end_y,
        receiver_kind=args.receiver_kind,
        receiver_side=args.receiver_side,
        receiver_center=args.receiver_center,
        receiver_length=args.receiver_length,
        receiver_array_start=args.receiver_array_start,
        receiver_array_end=args.receiver_array_end,
        receiver_array_num_elements=args.receiver_array_num_elements,
        receiver_array_element_length=args.receiver_array_element_length,
        receiver_array_start_x=args.receiver_array_start_x,
        receiver_array_start_y=args.receiver_array_start_y,
        receiver_array_end_x=args.receiver_array_end_x,
        receiver_array_end_y=args.receiver_array_end_y,
        receiver_sigma=args.receiver_sigma,
        receiver_array_geometry=args.receiver_array_geometry,
        receiver_box_xmin=args.receiver_box_xmin,
        receiver_box_xmax=args.receiver_box_xmax,
        receiver_box_ymin=args.receiver_box_ymin,
        receiver_box_ymax=args.receiver_box_ymax,
        receiver_box_left_num_elements=args.receiver_box_left_num_elements,
        receiver_box_top_num_elements=args.receiver_box_top_num_elements,
        receiver_box_right_num_elements=args.receiver_box_right_num_elements,
        export_receiver_vector_array=args.export_receiver_vector_array,
        auto_place_probes_inside_absorbing_layer=args.auto_place_probes_inside_absorbing_layer,
        source_segment_xmin=args.source_segment_xmin,
        source_segment_xmax=args.source_segment_xmax,
        receiver_segment_xmin=args.receiver_segment_xmin,
        receiver_segment_xmax=args.receiver_segment_xmax,
        material_kind=args.material_kind,
        defect_center_x=args.defect_center_x,
        defect_center_y=args.defect_center_y,
        defect_radius=args.defect_radius,
        contrast_factor=args.contrast_factor,
        defects_json=args.defects_json,
        defect_background_factor=args.defect_background_factor,
        defect_conflict_policy=args.defect_conflict_policy,
        theta=args.theta,
        dt=args.dt,
        num_steps=args.num_steps,
        source_frequency=args.source_frequency,
        every_k=args.every_k,
        visual_output_format=args.visual_output_format,
        export_material_fields=not args.no_export_material_fields,
        export_mesh=not args.no_export_mesh,
        export_sampling_conditions=not args.no_export_sampling_conditions,
        reuse_effective_solver=args.reuse_effective_solver,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_max_it=args.ksp_max_it,
        comm=MPI.COMM_WORLD,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
