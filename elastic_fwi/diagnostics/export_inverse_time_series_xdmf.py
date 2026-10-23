from __future__ import annotations

from elastic_fwi.observations.internal_gaussian_array import (
    internal_gaussian_array_observation,
)

import argparse
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

from dolfinx import fem, io

from elastic_fwi.diagnostics.export_direct_time_series_xdmf import (
    _build_direct_config,
    _export_material_fields_xdmf,
    _export_mesh_xdmf,
    _export_sampling_conditions_json,
    _export_static_function_vtk,
    _ranked_path,
    _snapshot_to_function,
)
from elastic_fwi.geometry.boundary_tags import build_boundary_meshtags
from elastic_fwi.inversion.adjoint_observation_sources import (
    assemble_full_adjoint_observation_rhs,
)
from elastic_fwi.inversion.material_imaging import (
    compute_material_qualitative_dof_image,
)
from elastic_fwi.inversion.material_parameter_gradients import (
    compute_material_parameter_correlation_kernels_dg0,
    compute_material_parameter_gradients_dg0,
)
from elastic_fwi.inversion.direct_adjoint_interaction_image import (
    compute_direct_adjoint_interaction_image_dg0,
)
from elastic_fwi.inversion.residual_wave_interaction_image import (
    compute_residual_wave_interaction_image_dg0,
    make_single_source_vector_array_full_rhs,
)
from elastic_fwi.inversion.newmark_adjoint import solve_newmark_discrete_adjoint_state
from elastic_fwi.inversion.observation_functionals import ObservationFunctionalWeights
from elastic_fwi.io import (
    build_vector_array_export_manifest,
    export_vector_array_residual_pair_csv,
    export_vector_array_residual_pair_npz,
    export_vector_array_time_series_npz,
    write_vector_array_manifest,
)
from elastic_fwi.observations import vector_array_observation
from elastic_fwi.pipelines.vector_array_observation_comparison import (
    run_vector_array_observation_comparison_from_direct_results,
)
from elastic_fwi.diagnostics.plot_vector_array_residuals import (
    plot_vector_array_residuals,
)
from elastic_fwi.transducers import build_line_transducer_array, build_polyline_transducer_array
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    _build_absorbing_layer_if_requested,
    _build_materials,
    _build_source_receiver_boundary_specs,
)
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    run_direct_observation_comparison,
)
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value



def _hash_array_for_mesh_contract(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.view(np.uint8).tobytes())
    return digest.hexdigest()


def _mesh_partition_fingerprint(msh, comm: MPI.Comm) -> dict[str, Any]:
    tdim = int(msh.topology.dim)
    msh.topology.create_connectivity(tdim, 0)
    connectivity = msh.topology.connectivity(tdim, 0)
    if connectivity is None:
        raise RuntimeError("Cell-to-vertex connectivity is unavailable.")

    vertex_map = msh.topology.index_map(0)
    cell_map = msh.topology.index_map(tdim)
    local = {
        "rank": int(comm.rank),
        "owned_vertices": int(vertex_map.size_local),
        "ghost_vertices": int(vertex_map.num_ghosts),
        "owned_cells": int(cell_map.size_local),
        "ghost_cells": int(cell_map.num_ghosts),
        "local_geometry_points": int(msh.geometry.x.shape[0]),
        "coordinates_sha256": _hash_array_for_mesh_contract(
            np.asarray(msh.geometry.x)
        ),
        "connectivity_array_sha256": _hash_array_for_mesh_contract(
            np.asarray(connectivity.array, dtype=np.int64)
        ),
        "connectivity_offsets_sha256": _hash_array_for_mesh_contract(
            np.asarray(connectivity.offsets, dtype=np.int64)
        ),
    }

    rows = comm.gather(local, root=0)
    result: dict[str, Any] | None = None
    if comm.rank == 0:
        canonical = json.dumps(
            rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        result = {
            "global_owned_vertices": int(
                sum(row["owned_vertices"] for row in rows)
            ),
            "global_owned_cells": int(
                sum(row["owned_cells"] for row in rows)
            ),
            "rank_ordered_partition_sha256": hashlib.sha256(
                canonical
            ).hexdigest(),
            "per_rank": rows,
        }

    result = comm.bcast(result, root=0)
    assert result is not None
    return result

def _record_step_time(record: Any, *, fallback_index: int, dt: float) -> tuple[int, float]:
    if isinstance(record, dict):
        return int(record["step"]), float(record["time"])
    if record is not None and hasattr(record, "step") and hasattr(record, "time"):
        return int(record.step), float(record.time)
    step = int(fallback_index + 1)
    return step, float(step * dt)


def _select_vector_time_series_items(
    *,
    snapshots,
    time_records,
    dt: float,
    every_k: int,
    export_initial_state: bool,
    export_final_state: bool,
    initial_snapshot,
) -> list[tuple[int, float, Any]]:
    """Select physical-time export items without relabelling solver states.

    Direct and adjoint snapshot tuples contain the states at steps 1..N.  The
    explicit initial state, when requested, is inserted at step 0 and time 0.
    The last computed state is always retained when ``export_final_state`` is
    enabled, even when its step is not a multiple of ``every_k``.
    """
    snapshots = tuple(snapshots)
    records = None if time_records is None else tuple(time_records)

    if records is not None and len(records) != len(snapshots):
        raise ValueError("time_records and snapshots must have the same length.")

    selected: list[tuple[int, float, Any]] = []
    if export_initial_state:
        selected.append((0, 0.0, initial_snapshot))

    selected_snapshot_indices: set[int] = set()
    for index, snapshot in enumerate(snapshots):
        record = None if records is None else records[index]
        step, time = _record_step_time(record, fallback_index=index, dt=dt)
        if step % every_k == 0:
            selected.append((step, time, snapshot))
            selected_snapshot_indices.add(index)

    if export_final_state and snapshots:
        final_index = len(snapshots) - 1
        if final_index not in selected_snapshot_indices:
            record = None if records is None else records[final_index]
            step, time = _record_step_time(record, fallback_index=final_index, dt=dt)
            selected.append((step, time, snapshots[final_index]))

    return selected


def _snapshot_or_zero_to_function(*, snapshot, template, field_name: str):
    if snapshot is None:
        field = fem.Function(template.function_space)
        field.name = field_name
        field.x.array[:] = 0.0
        field.x.scatter_forward()
        return field
    return _snapshot_to_function(
        snapshot=snapshot,
        template=template,
        name=field_name,
    )


def _export_vector_time_series_vtk(
    *,
    output_path: Path,
    V,
    snapshots,
    dt: float,
    every_k: int,
    field_name: str,
    comm: MPI.Comm,
    time_records=None,
    export_initial_state: bool = False,
    export_final_state: bool = True,
    initial_snapshot=None,
) -> dict[str, Any]:
    if every_k <= 0:
        raise ValueError("every_k must be strictly positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = _select_vector_time_series_items(
        snapshots=snapshots,
        time_records=time_records,
        dt=dt,
        every_k=every_k,
        export_initial_state=export_initial_state,
        export_final_state=export_final_state,
        initial_snapshot=initial_snapshot,
    )

    exported_indices: list[int] = []
    exported_times: list[float] = []
    template = fem.Function(V)
    template.name = field_name

    with io.VTKFile(comm, str(output_path), "w") as vtk:
        for step, time, snapshot in selected:
            field = _snapshot_or_zero_to_function(
                snapshot=snapshot,
                template=template,
                field_name=field_name,
            )
            vtk.write_function(field, time)
            exported_indices.append(int(step))
            exported_times.append(float(time))

    final_time = None if not exported_times else float(exported_times[-1])
    return {
        "status": "ok",
        "pvd": str(output_path),
        "field_name": field_name,
        "num_snapshots_available": int(len(tuple(snapshots))),
        "num_snapshots_exported": int(len(exported_indices)),
        "exported_indices": exported_indices,
        "exported_times": exported_times,
        "volume_sampling_stride": int(every_k),
        "volume_sampling_period": float(every_k * dt),
        "export_initial_state_requested": bool(export_initial_state),
        "export_final_state_requested": bool(export_final_state),
        "initial_state_exported": bool(exported_indices and exported_indices[0] == 0),
        "final_exported_time": final_time,
        "paraview_entrypoint": str(output_path),
    }


def _export_vector_time_series_xdmf(
    *,
    output_path: Path,
    V,
    snapshots,
    dt: float,
    every_k: int,
    field_name: str,
    comm: MPI.Comm,
    export_xdmf: bool = False,
    time_records=None,
    export_initial_state: bool = False,
    export_final_state: bool = True,
    initial_snapshot=None,
) -> dict[str, Any]:
    if every_k <= 0:
        raise ValueError("every_k must be strictly positive.")

    selected = _select_vector_time_series_items(
        snapshots=snapshots,
        time_records=time_records,
        dt=dt,
        every_k=every_k,
        export_initial_state=export_initial_state,
        export_final_state=export_final_state,
        initial_snapshot=initial_snapshot,
    )

    vtk_export = _export_vector_time_series_vtk(
        output_path=output_path.parent / "vtk" / f"{output_path.stem}.pvd",
        V=V,
        snapshots=snapshots,
        dt=dt,
        every_k=every_k,
        field_name=field_name,
        comm=comm,
        time_records=time_records,
        export_initial_state=export_initial_state,
        export_final_state=export_final_state,
        initial_snapshot=initial_snapshot,
    )

    if export_xdmf:
        template = fem.Function(V)
        template.name = field_name
        with io.XDMFFile(comm, str(output_path), "w") as xdmf:
            xdmf.write_mesh(V.mesh)
            for _, time, snapshot in selected:
                field = _snapshot_or_zero_to_function(
                    snapshot=snapshot,
                    template=template,
                    field_name=field_name,
                )
                xdmf.write_function(field, time)

    return {
        "status": "ok",
        "xdmf": str(output_path) if export_xdmf else None,
        "h5": str(output_path.with_suffix(".h5")) if export_xdmf else None,
        "pvd": vtk_export["pvd"],
        "vtk": vtk_export,
        "field_name": field_name,
        "num_snapshots_available": vtk_export["num_snapshots_available"],
        "num_snapshots_exported": vtk_export["num_snapshots_exported"],
        "exported_indices": vtk_export["exported_indices"],
        "exported_times": vtk_export["exported_times"],
        "volume_sampling_stride": int(every_k),
        "volume_sampling_period": float(every_k * dt),
        "export_initial_state_requested": bool(export_initial_state),
        "export_final_state_requested": bool(export_final_state),
        "initial_state_exported": vtk_export["initial_state_exported"],
        "final_exported_time": vtk_export["final_exported_time"],
        "paraview_entrypoint": vtk_export["pvd"],
        "paraview_entrypoint_xdmf": str(output_path) if export_xdmf else None,
    }

def _export_gradient_proxy_xdmf_and_npz(
    *,
    output_dir: Path,
    V,
    direct_snapshots,
    adjoint_vectors,
    comm: MPI.Comm,
    export_xdmf: bool = False,
) -> dict[str, Any]:
    image = compute_material_qualitative_dof_image(
        direct_snapshots=direct_snapshots,
        equation_adjoint_vectors=adjoint_vectors,
        function_space=V,
        comm=comm,
        absolute_value=True,
        normalize=True,
    )

    xdmf_path = output_dir / "material_gradient_proxy.xdmf"
    npz_path = _ranked_path(output_dir / "material_gradient_proxy.npz", comm)

    template = fem.Function(V)
    template.name = "material_gradient_proxy"

    gradient_field = _snapshot_to_function(
        snapshot=np.asarray(image.local_values, dtype=float),
        template=template,
        name="material_gradient_proxy",
    )

    if export_xdmf:
        with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
            xdmf.write_mesh(V.mesh)
            xdmf.write_function(gradient_field, 0.0)

    vtk_export = _export_static_function_vtk(
        output_path=output_dir / "vtk" / "material_gradient_proxy.pvd",
        function=gradient_field,
        comm=comm,
        time=0.0,
    )

    np.savez(
        npz_path,
        kind=np.asarray("material_gradient_proxy", dtype=np.str_),
        mpi_rank=np.asarray(comm.rank, dtype=np.int64),
        mpi_size=np.asarray(comm.size, dtype=np.int64),
        local_values=np.asarray(image.local_values, dtype=np.float64),
        local_coordinates=(
            np.empty((0, 2), dtype=np.float64)
            if image.local_coordinates is None
            else np.asarray(image.local_coordinates, dtype=np.float64)
        ),
        global_min=np.asarray(image.global_min, dtype=np.float64),
        global_max=np.asarray(image.global_max, dtype=np.float64),
        global_l2_norm=np.asarray(image.global_l2_norm, dtype=np.float64),
        global_sum=np.asarray(image.global_sum, dtype=np.float64),
        normalized=np.asarray(image.normalized, dtype=np.bool_),
    )

    return {
        "status": "ok",
        "xdmf": str(xdmf_path) if export_xdmf else None,
        "h5": str(xdmf_path.with_suffix(".h5")) if export_xdmf else None,
        "npz": str(npz_path),
        "vtk": vtk_export,
        "field_name": "material_gradient_proxy",
        "scientific_role": "diagnostic_proxy_not_primary_gradient",
        "num_steps": int(image.num_steps),
        "global_min": float(image.global_min),
        "global_max": float(image.global_max),
        "global_l2_norm": float(image.global_l2_norm),
        "global_sum": float(image.global_sum),
        "normalized": bool(image.normalized),
        "paraview_entrypoint": vtk_export["pvd"],
        "paraview_entrypoint_xdmf": str(xdmf_path) if export_xdmf else None,
    }


def _export_direct_adjoint_interaction_image_xdmf(
    *,
    output_dir: Path,
    V,
    direct_snapshots,
    adjoint_snapshots,
    dt: float,
    theta: float,
    comm: MPI.Comm,
    export_xdmf: bool = False,
) -> dict[str, Any]:
    result = compute_direct_adjoint_interaction_image_dg0(
        V=V,
        direct_snapshots=direct_snapshots,
        adjoint_snapshots=adjoint_snapshots,
        dt=dt,
    )

    xdmf_path = output_dir / "direct_adjoint_interaction_image.xdmf"

    if export_xdmf:
        with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
            xdmf.write_mesh(V.mesh)
            for field in result.fields.values():
                xdmf.write_function(field, 0.0)

    vtk_exports = {}
    for name, field in result.fields.items():
        vtk_exports[name] = _export_static_function_vtk(
            output_path=output_dir / "vtk" / f"direct_adjoint_interaction_{name}.pvd",
            function=field,
            comm=comm,
            time=0.0,
        )

    default_field = (
        "log_interaction_image"
        if "log_interaction_image" in vtk_exports
        else sorted(vtk_exports)[0]
    )

    return {
        "status": "ok",
        "xdmf": str(xdmf_path) if export_xdmf else None,
        "h5": str(xdmf_path.with_suffix(".h5")) if export_xdmf else None,
        "vtk": vtk_exports,
        "field_names": sorted(result.fields.keys()),
        "metadata": result.metadata,
        "paraview_entrypoint": vtk_exports[default_field]["pvd"],
        "paraview_entrypoint_xdmf": str(xdmf_path) if export_xdmf else None,
    }



def _export_residual_wave_interaction_image_xdmf(
    *,
    output_dir: Path,
    V,
    reflection_residual_wave_snapshots,
    transmission_residual_wave_snapshots,
    dt: float,
    comm: MPI.Comm,
    export_xdmf: bool = False,
) -> dict[str, Any]:
    result = compute_residual_wave_interaction_image_dg0(
        V=V,
        reflection_residual_wave_snapshots=reflection_residual_wave_snapshots,
        transmission_residual_wave_snapshots=transmission_residual_wave_snapshots,
        dt=dt,
    )

    xdmf_path = output_dir / "residual_wave_interaction_image.xdmf"

    if export_xdmf:
        with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
            xdmf.write_mesh(V.mesh)
            for field in result.fields.values():
                xdmf.write_function(field, 0.0)

    vtk_exports: dict[str, Any] = {}
    for name, field in result.fields.items():
        vtk_exports[name] = _export_static_function_vtk(
            output_path=output_dir / "vtk" / f"residual_wave_interaction_{name}.pvd",
            function=field,
            comm=comm,
            time=0.0,
        )

    default_field = (
        "log_interaction_image"
        if "log_interaction_image" in vtk_exports
        else sorted(vtk_exports)[0]
    )

    return {
        "status": "ok",
        "xdmf": str(xdmf_path) if export_xdmf else None,
        "h5": str(xdmf_path.with_suffix(".h5")) if export_xdmf else None,
        "vtk": vtk_exports,
        "field_names": sorted(result.fields.keys()),
        "metadata": result.metadata,
        "paraview_entrypoint": vtk_exports[default_field]["pvd"],
        "paraview_entrypoint_xdmf": str(xdmf_path) if export_xdmf else None,
        "scientific_role": "residual_wave_interaction_image",
    }




def _apply_gradient_export_contract(
    payload: dict[str, Any],
    *,
    gradient_export_label: str,
    absolute_value: bool,
    normalize: bool,
    residual_wave: bool = False,
) -> dict[str, Any]:
    semantics = _gradient_export_semantics(gradient_export_label)

    metadata = dict(payload.get("metadata", {}))
    metadata["absolute_value"] = bool(absolute_value)
    metadata["absolute_value_stage"] = (
        "after_time_integration" if bool(absolute_value) else "not_applied"
    )
    metadata["normalize"] = bool(normalize)
    metadata["gradient_export_label"] = str(gradient_export_label)
    metadata["export_semantics"] = semantics

    out = dict(payload)
    out["metadata"] = metadata
    out["gradient_export_label"] = str(gradient_export_label)
    out["export_semantics"] = semantics

    if str(gradient_export_label) == "article_safe":
        role = "residual_wave_interaction_indicator" if residual_wave else "normalized_qualitative_sensitivity_image"
    elif str(gradient_export_label) == "raw_signed":
        role = "signed_non_normalized_residual_wave_interaction_density" if residual_wave else "signed_non_normalized_material_gradient_density"
    else:
        raise ValueError(
            "gradient_export_label must be one of 'article_safe' or 'raw_signed'."
        )

    out["scientific_role"] = role

    return out

def _export_residual_wave_material_parameter_gradients_xdmf(
    *,
    output_dir: Path,
    V,
    reflection_residual_wave_snapshots,
    transmission_residual_wave_snapshots,
    dt: float,
    comm: MPI.Comm,
    export_xdmf: bool = False,
    absolute_value: bool = True,
    normalize: bool = True,
    gradient_export_label: str = "article_safe",
) -> dict[str, Any]:
    result = compute_material_parameter_correlation_kernels_dg0(
        V=V,
        direct_snapshots=reflection_residual_wave_snapshots,
        adjoint_snapshots=transmission_residual_wave_snapshots,
        dt=dt,
        absolute_value=absolute_value,
        normalize=normalize,
        field_prefix="kernel",
    )

    xdmf_path = output_dir / "residual_wave_material_parameter_kernels.xdmf"

    renamed_fields = {}
    for name, field in result.fields.items():
        if name.startswith("kernel_"):
            renamed_name = "residual_interaction_" + name.removeprefix("kernel_")
        else:
            renamed_name = "residual_interaction_" + name
        field.name = renamed_name
        renamed_fields[renamed_name] = field

    if export_xdmf:
        with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
            xdmf.write_mesh(V.mesh)
            for field in result.fields.values():
                xdmf.write_function(field, 0.0)

    vtk_exports: dict[str, Any] = {}
    for name, field in renamed_fields.items():
        vtk_exports[name] = _export_static_function_vtk(
            output_path=output_dir / "vtk" / f"{name}.pvd",
            function=field,
            comm=comm,
            time=0.0,
        )

    default_field = (
        "residual_interaction_rho"
        if "residual_interaction_rho" in vtk_exports
        else sorted(vtk_exports)[0]
    )

    metadata = {
        **result.metadata,
        "kind": "residual_wave_material_parameter_kernels_dg0",
        "imaging_mode": "residual_wave_interaction",
        "first_wavefield": "reflection_residual_wavefield",
        "second_wavefield": "transmission_residual_wavefield",
        "field_prefix": "residual_interaction",
        "mathematical_role": "residual_wave_interaction_indicator",
        # Both booleans below are scoped to the *material model* m, which is the
        # default reading in an FWI context: K^RI is neither the gradient of, nor
        # a descent direction for, the transmitted-only misfit in m.  In the
        # *perturbation* variable at a fixed reference model the status differs
        # and is stated by `variational_status_in_perturbation` below.  Read
        # together, both statements agree with the theorem of the accompanying article.
        "is_true_discrete_gradient": False,
        "is_descent_direction": False,
        "descent_direction_scope": "material_model_m",
        "variational_status_in_perturbation": (
            "exact derivative density at the origin of a linearised "
            "least-squares functional Q with a back-propagated reflection "
            "residual as illuminating field; genuine gradient of Q in the "
            "discrete DG0 space; -K^RI is a descent direction for Q in the "
            "directional-derivative sense"
        ),
        "gradient_export_label": str(gradient_export_label),
    }

    return {
        "status": "ok",
        "xdmf": str(xdmf_path) if export_xdmf else None,
        "h5": str(xdmf_path.with_suffix(".h5")) if export_xdmf else None,
        "vtk": vtk_exports,
        "field_names": sorted(renamed_fields.keys()),
        "metadata": metadata,
        "paraview_entrypoint": vtk_exports[default_field]["pvd"],
        "paraview_entrypoint_xdmf": str(xdmf_path) if export_xdmf else None,
        "scientific_role": "residual_wave_interaction_indicator",
        "gradient_export_label": str(gradient_export_label),
    }


def _gradient_export_semantics(gradient_export_label: str) -> dict[str, Any]:
    label = str(gradient_export_label).lower()

    if label == "article_safe":
        return {
            "kind": "qualitative_image_export",
            "scientific_role": "normalized_qualitative_sensitivity_image",
            "manuscript_label": "qualitative image",
            "description": (
                "Absolute-value, globally normalized material sensitivity-density "
                "or residual-interaction map. The absolute value is applied "
                "after temporal integration."
            ),
            "signed": False,
            "absolute_value": True,
            "absolute_value_stage": "after_time_integration",
            "normalize": True,
            "recommended_usage": (
                "Use for qualitative localization/visualization only; do not "
                "describe this export as a raw signed gradient."
            ),
            "manuscript_do_not_call": "raw gradient",
        }

    if label == "raw_signed":
        return {
            "kind": "signed_gradient_density_export",
            "scientific_role": "signed_non_normalized_material_gradient_density",
            "manuscript_label": "signed non-normalized gradient density",
            "description": (
                "Signed, non-normalized material-gradient-density export. "
                "No absolute value and no global normalization are applied."
            ),
            "signed": True,
            "absolute_value": False,
            "absolute_value_stage": "not_applied",
            "normalize": False,
            "recommended_usage": (
                "Use for mathematical diagnostics of signed gradient "
                "information; do not confuse with the qualitative article image."
            ),
            "manuscript_do_not_call": "absolute-value qualitative image",
        }

    raise ValueError(
        "gradient_export_label must be one of 'article_safe' or 'raw_signed'."
    )


def _export_material_parameter_gradients_xdmf(
    *,
    output_dir: Path,
    V,
    direct_snapshots,
    adjoint_snapshots,
    dt: float,
    theta: float,
    comm: MPI.Comm,
    export_xdmf: bool = False,
    absolute_value: bool = True,
    normalize: bool = True,
    gradient_export_label: str = "article_safe",
) -> dict[str, Any]:
    semantics = _gradient_export_semantics(gradient_export_label)
    result = compute_material_parameter_gradients_dg0(
        V=V,
        direct_snapshots=direct_snapshots,
        adjoint_snapshots=adjoint_snapshots,
        dt=dt,
        theta=theta,
        absolute_value=absolute_value,
        normalize=normalize,
    )

    xdmf_path = output_dir / "material_parameter_gradients.xdmf"

    if export_xdmf:
        with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
            xdmf.write_mesh(V.mesh)
            for field in result.fields.values():
                xdmf.write_function(field, 0.0)

    vtk_exports: dict[str, Any] = {}
    for name, field in result.fields.items():
        vtk_exports[name] = _export_static_function_vtk(
            output_path=output_dir / "vtk" / f"material_parameter_gradients_{name}.pvd",
            function=field,
            comm=comm,
            time=0.0,
        )

    default_field = "gradient_rho" if "gradient_rho" in vtk_exports else sorted(vtk_exports)[0]
    result.metadata["export_semantics"] = semantics

    return _apply_gradient_export_contract(
        {
            "status": "ok",
            "xdmf": str(xdmf_path) if export_xdmf else None,
            "h5": str(xdmf_path.with_suffix(".h5")) if export_xdmf else None,
            "vtk": vtk_exports,
            "field_names": sorted(result.fields.keys()),
            "metadata": result.metadata,
            "paraview_entrypoint": vtk_exports[default_field]["pvd"],
            "paraview_entrypoint_xdmf": str(xdmf_path) if export_xdmf else None,
        },
        gradient_export_label=str(gradient_export_label),
        absolute_value=absolute_value,
        normalize=normalize,
        residual_wave=False,
    )


def _gradient_export_specs(gradient_export_mode: str) -> list[dict[str, Any]]:
    mode = str(gradient_export_mode).lower()
    if mode not in {"article_safe", "raw_signed", "both"}:
        raise ValueError(
            "gradient_export_mode must be one of "
            "'article_safe', 'raw_signed', or 'both'."
        )

    article_safe = {
        "manifest_key": "gradient_combinations",
        "subdir": "gradient_combinations",
        "label": "article_safe",
        "absolute_value": True,
        "normalize": True,
        "semantics": _gradient_export_semantics("article_safe"),
    }
    raw_signed = {
        "manifest_key": "gradient_combinations_raw_signed" if mode == "both" else "gradient_combinations",
        "subdir": "gradient_combinations_raw_signed",
        "label": "raw_signed",
        "absolute_value": False,
        "normalize": False,
        "semantics": _gradient_export_semantics("raw_signed"),
    }

    if mode == "article_safe":
        return [article_safe]
    if mode == "raw_signed":
        return [raw_signed]
    return [article_safe, raw_signed]


def _export_gradient_combination_set(
    *,
    output_dir: Path,
    V,
    comparison,
    adjoint,
    reflection_residual_wave,
    transmission_residual_wave,
    dt: float,
    theta: float,
    comm: MPI.Comm,
    export_xdmf: bool,
    absolute_value: bool,
    normalize: bool,
    gradient_export_label: str,
) -> dict[str, Any]:
    gradient_combinations_export: dict[str, Any] = {
        "classical_full": _export_material_parameter_gradients_xdmf(
            output_dir=output_dir / "classical_full",
            V=V,
            direct_snapshots=comparison.simulated.displacement_snapshots,
            adjoint_snapshots=adjoint.equation_adjoint_vectors,
            dt=dt,
            theta=theta,
            comm=comm,
            export_xdmf=export_xdmf,
            absolute_value=absolute_value,
            normalize=normalize,
            gradient_export_label=gradient_export_label,
        )
    }

    if reflection_residual_wave is not None and transmission_residual_wave is not None:
        gradient_combinations_export["standard_reflection_only"] = (
            _export_material_parameter_gradients_xdmf(
                output_dir=output_dir / "standard_reflection_only",
                V=V,
                direct_snapshots=comparison.simulated.displacement_snapshots,
                adjoint_snapshots=reflection_residual_wave.equation_adjoint_vectors,
                dt=dt,
                theta=theta,
                comm=comm,
                export_xdmf=export_xdmf,
                absolute_value=absolute_value,
                normalize=normalize,
                gradient_export_label=gradient_export_label,
            )
        )

        gradient_combinations_export["standard_transmission_only"] = (
            _export_material_parameter_gradients_xdmf(
                output_dir=output_dir / "standard_transmission_only",
                V=V,
                direct_snapshots=comparison.simulated.displacement_snapshots,
                adjoint_snapshots=transmission_residual_wave.equation_adjoint_vectors,
                dt=dt,
                theta=theta,
                comm=comm,
                export_xdmf=export_xdmf,
                absolute_value=absolute_value,
                normalize=normalize,
                gradient_export_label=gradient_export_label,
            )
        )

        gradient_combinations_export["residual_interaction"] = (
            _export_residual_wave_material_parameter_gradients_xdmf(
                output_dir=output_dir / "residual_interaction",
                V=V,
                reflection_residual_wave_snapshots=reflection_residual_wave.equation_adjoint_vectors,
                transmission_residual_wave_snapshots=transmission_residual_wave.equation_adjoint_vectors,
                dt=dt,
                comm=comm,
                export_xdmf=export_xdmf,
                absolute_value=absolute_value,
                normalize=normalize,
                gradient_export_label=gradient_export_label,
            )
        )

    return {
        name: _apply_gradient_export_contract(
            payload,
            gradient_export_label=str(gradient_export_label),
            absolute_value=absolute_value,
            normalize=normalize,
            residual_wave=(name == "residual_interaction"),
        )
        for name, payload in gradient_combinations_export.items()
    }


def _export_residuals_csv_png(
    *,
    output_dir: Path,
    residuals,
    comm: MPI.Comm,
) -> dict[str, Any]:
    csv_path = _ranked_path(output_dir / "observation_residuals.csv", comm)
    png_path = output_dir / "observation_residuals.png"

    if comm.rank == 0:
        import csv
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "time",
                    "reflection_residual",
                    "transmission_residual",
                ],
            )
            writer.writeheader()

            for t, r1, r2 in zip(
                residuals.times,
                residuals.reflection.values,
                residuals.transmission.values,
                strict=True,
            ):
                writer.writerow(
                    {
                        "time": float(t),
                        "reflection_residual": float(r1),
                        "transmission_residual": float(r2),
                    }
                )

        plt.figure()
        plt.plot(residuals.times, residuals.reflection.values, label="reflection residual")
        plt.plot(residuals.times, residuals.transmission.values, label="transmission residual")
        plt.xlabel("time")
        plt.ylabel("residual")
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

    return {
        "status": "ok",
        "csv": str(csv_path),
        "png": str(png_path),
        "num_steps": int(residuals.num_steps),
        "final_time": float(residuals.final_time),
        "max_abs_residual": float(residuals.max_abs_residual),
    }



def _side_to_y(
    *,
    side: str,
    domain_length_y: float,
    absorbing_layer_width: float | None,
    auto_place_probes_inside_absorbing_layer: bool,
) -> float:
    side = str(side)
    if auto_place_probes_inside_absorbing_layer and absorbing_layer_width is not None:
        if side == "bottom":
            return float(absorbing_layer_width)
        if side == "top":
            return float(domain_length_y - absorbing_layer_width)

    if side == "bottom":
        return 0.0
    if side == "top":
        return float(domain_length_y)

    raise ValueError(
        "vector_array observation currently expects horizontal bottom/top arrays."
    )


def _resolve_vector_array_receiver_sigma(
    *,
    config: DirectObservationPipelineConfig,
    receiver_sigma: float | None,
) -> float:
    if receiver_sigma is not None:
        sigma = float(receiver_sigma)
    elif config.receiver_array_element_length is not None:
        sigma = float(config.receiver_array_element_length)
    elif config.source_array_element_length is not None:
        sigma = float(config.source_array_element_length)
    else:
        sigma = float(config.source_sigma)

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("receiver_sigma must be strictly positive.")

    return sigma


def _build_vector_array_observation_specs_from_config(
    *,
    config: DirectObservationPipelineConfig,
    receiver_sigma: float | None,
    receiver_array_geometry: str = "line",
    receiver_box_xmin: float | None = None,
    receiver_box_xmax: float | None = None,
    receiver_box_ymin: float | None = None,
    receiver_box_ymax: float | None = None,
    receiver_box_left_num_elements: int = 13,
    receiver_box_top_num_elements: int = 11,
    receiver_box_right_num_elements: int = 13,
):
    sigma = _resolve_vector_array_receiver_sigma(
        config=config,
        receiver_sigma=receiver_sigma,
    )

    source_start_x = (
        config.source_array_start_x
        if config.source_array_start_x is not None
        else config.source_segment_xmin
    )
    source_end_x = (
        config.source_array_end_x
        if config.source_array_end_x is not None
        else config.source_segment_xmax
    )
    source_start_y = (
        config.source_array_start_y
        if config.source_array_start_y is not None
        else _side_to_y(
            side=config.source_side,
            domain_length_y=config.domain_length_y,
            absorbing_layer_width=config.absorbing_layer_width,
            auto_place_probes_inside_absorbing_layer=config.auto_place_probes_inside_absorbing_layer,
        )
    )
    source_end_y = (
        config.source_array_end_y
        if config.source_array_end_y is not None
        else source_start_y
    )

    receiver_start_x = (
        config.receiver_array_start_x
        if config.receiver_array_start_x is not None
        else config.receiver_segment_xmin
    )
    receiver_end_x = (
        config.receiver_array_end_x
        if config.receiver_array_end_x is not None
        else config.receiver_segment_xmax
    )
    receiver_start_y = (
        config.receiver_array_start_y
        if config.receiver_array_start_y is not None
        else _side_to_y(
            side=config.receiver_side,
            domain_length_y=config.domain_length_y,
            absorbing_layer_width=config.absorbing_layer_width,
            auto_place_probes_inside_absorbing_layer=config.auto_place_probes_inside_absorbing_layer,
        )
    )
    receiver_end_y = (
        config.receiver_array_end_y
        if config.receiver_array_end_y is not None
        else receiver_start_y
    )

    gamma1 = build_line_transducer_array(
        name="gamma1",
        role="source_reflection",
        start_x=float(source_start_x),
        start_y=float(source_start_y),
        end_x=float(source_end_x),
        end_y=float(source_end_y),
        num_elements=int(config.source_array_num_elements),
        sigma=sigma,
        tag=1,
    )

    receiver_array_geometry = str(receiver_array_geometry)

    if receiver_array_geometry == "line":
        gamma2 = build_line_transducer_array(
            name="gamma2",
            role="transmission",
            start_x=float(receiver_start_x),
            start_y=float(receiver_start_y),
            end_x=float(receiver_end_x),
            end_y=float(receiver_end_y),
            num_elements=int(config.receiver_array_num_elements),
            sigma=sigma,
            tag=2,
        )
    elif receiver_array_geometry == "box_except_gamma1":
        xmin = float(receiver_box_xmin if receiver_box_xmin is not None else receiver_start_x)
        xmax = float(receiver_box_xmax if receiver_box_xmax is not None else receiver_end_x)
        ymin = float(receiver_box_ymin if receiver_box_ymin is not None else source_start_y)
        ymax = float(receiver_box_ymax if receiver_box_ymax is not None else receiver_start_y)

        if not (xmin < xmax):
            raise ValueError("receiver_box_xmin must be < receiver_box_xmax.")
        if not (ymin < ymax):
            raise ValueError("receiver_box_ymin must be < receiver_box_ymax.")

        gamma2 = build_polyline_transducer_array(
            name="gamma2",
            role="transmission",
            vertices=[
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
            ],
            num_elements_per_segment=[
                int(receiver_box_left_num_elements),
                int(receiver_box_top_num_elements),
                int(receiver_box_right_num_elements),
            ],
            sigma=sigma,
            tag=2,
        )
    else:
        raise ValueError(
            "receiver_array_geometry must be either 'line' or 'box_except_gamma1'."
        )

    return (
        vector_array_observation(array=gamma1),
        vector_array_observation(array=gamma2),
        sigma,
    )


def _array_side_labels(series) -> list[str]:
    centers = np.asarray(series.spec.centers, dtype=float)
    array = series.spec.array

    if array.role in {"source", "reflection", "source_reflection"}:
        return ["gamma1"] * int(centers.shape[0])

    if array.centers_override is None:
        return ["top"] * int(centers.shape[0])

    xmin = float(np.min(centers[:, 0]))
    xmax = float(np.max(centers[:, 0]))
    ymax = float(np.max(centers[:, 1]))
    scale = max(1.0, abs(xmin), abs(xmax), abs(ymax))
    atol = 1.0e-12 * scale
    labels: list[str] = []
    for x, y in centers:
        if np.isclose(x, xmin, rtol=0.0, atol=atol):
            labels.append("left")
        elif np.isclose(y, ymax, rtol=0.0, atol=atol):
            labels.append("top")
        elif np.isclose(x, xmax, rtol=0.0, atol=atol):
            labels.append("right")
        else:
            labels.append("polyline")
    return labels


def _write_vector_array_coordinates_csv(*, path: Path, vector_comparison) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for series in (
        vector_comparison.simulated_reflection,
        vector_comparison.simulated_transmission,
    ):
        centers = np.asarray(series.spec.centers, dtype=float)
        sides = _array_side_labels(series)
        for receiver_id, ((x, y), side) in enumerate(zip(centers, sides)):
            rows.append(
                {
                    "array": series.spec.array.name,
                    "role": series.spec.role,
                    "receiver_id": int(receiver_id),
                    "side": side,
                    "x": float(x),
                    "y": float(y),
                    "sigma": float(series.spec.sigma),
                }
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["array", "role", "receiver_id", "side", "x", "y", "sigma"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _export_vector_array_artifacts(
    *,
    output_dir: Path,
    vector_comparison,
    receiver_sigma: float,
    comm: MPI.Comm,
) -> dict[str, Any]:
    vector_dir = output_dir / "vector_array"

    files: dict[str, Any] = {
        "status": "ok",
        "directory": str(vector_dir),
        "written_by_rank": 0,
    }

    if comm.rank == 0:
        vector_dir.mkdir(parents=True, exist_ok=True)

        files["signals"] = {
            "simulated_reflection": str(
                export_vector_array_time_series_npz(
                    path=vector_dir / "simulated_reflection_signals.npz",
                    series=vector_comparison.simulated_reflection,
                    label="simulated_reflection",
                )
            ),
            "observed_reflection": str(
                export_vector_array_time_series_npz(
                    path=vector_dir / "observed_reflection_signals.npz",
                    series=vector_comparison.observed_reflection,
                    label="observed_reflection",
                )
            ),
            "simulated_transmission": str(
                export_vector_array_time_series_npz(
                    path=vector_dir / "simulated_transmission_signals.npz",
                    series=vector_comparison.simulated_transmission,
                    label="simulated_transmission",
                )
            ),
            "observed_transmission": str(
                export_vector_array_time_series_npz(
                    path=vector_dir / "observed_transmission_signals.npz",
                    series=vector_comparison.observed_transmission,
                    label="observed_transmission",
                )
            ),
        }

        files["coordinates_csv"] = str(
            _write_vector_array_coordinates_csv(
                path=vector_dir / "vector_array_coordinates.csv",
                vector_comparison=vector_comparison,
            )
        )

        files["residual_npz"] = export_vector_array_residual_pair_npz(
            output_dir=vector_dir,
            residuals=vector_comparison.residuals,
        )

        files["residual_csv"] = str(
            export_vector_array_residual_pair_csv(
                path=vector_dir / "vector_array_residuals.csv",
                residuals=vector_comparison.residuals,
            )
        )

        residual_figures = plot_vector_array_residuals(
            reflection_npz=Path(files["residual_npz"]["reflection"]),
            transmission_npz=Path(files["residual_npz"]["transmission"]),
            output_dir=vector_dir / "figures",
        )

        files["residual_figures"] = residual_figures["outputs"]
        files["residual_plot_summary"] = residual_figures

        manifest = build_vector_array_export_manifest(
            residuals=vector_comparison.residuals,
            files=files,
            extra={
                "receiver_sigma": float(receiver_sigma),
                "adjoint_rhs": vector_comparison.adjoint_rhs.as_dict(),
                "comparison": vector_comparison.as_dict(),
            },
        )

        files["manifest"] = str(
            write_vector_array_manifest(
                path=vector_dir / "vector_array_manifest.json",
                payload=manifest,
            )
        )

    comm.Barrier()
    return files



def export_inverse_time_series_xdmf(
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
    observation_kind: str = "scalar_energy",
    source_kind: str = "boundary_segment",
    source_side: str = "bottom",
    source_center: float | None = None,
    source_length: float | None = None,
    source_array_start: float | None = None,
    source_array_end: float | None = None,
    source_array_num_elements: int = 1,
    source_array_element_length: float | None = None,
    source_center_x: float = 0.5,
    source_center_y: float = 0.5,
    source_radius: float = 0.05,
    source_sigma: float = 0.05,
    source_direction_x: float = 0.0,
    source_direction_y: float = 1.0,
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
    receiver_sigma: float | None = None,
    receiver_array_start_x: float | None = None,
    receiver_array_start_y: float | None = None,
    receiver_array_end_x: float | None = None,
    receiver_array_end_y: float | None = None,
    receiver_array_geometry: str = "line",
    receiver_box_xmin: float | None = None,
    receiver_box_xmax: float | None = None,
    receiver_box_ymin: float | None = None,
    receiver_box_ymax: float | None = None,
    receiver_box_left_num_elements: int = 13,
    receiver_box_top_num_elements: int = 11,
    receiver_box_right_num_elements: int = 13,
    auto_place_probes_inside_absorbing_layer: bool = False,
    source_segment_xmin: float = 0.0,
    source_segment_xmax: float = 1.0,
    receiver_segment_xmin: float = 0.0,
    receiver_segment_xmax: float = 1.0,
    simulated_material_kind: str = "healthy",
    observed_material_kind: str = "localized_disk_defect",
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
    source_num_periods: float = 1.0,
    reflection_weight: float = 1.0,
    transmission_weight: float = 1.0,
    every_k: int = 1,
    ksp_type: str = "cg",
    pc_type: str = "jacobi",
    ksp_rtol: float = 1.0e-10,
    ksp_atol: float = 1.0e-12,
    ksp_max_it: int = 1000,
    reuse_direct_effective_solver: bool = True,
    reuse_adjoint_effective_solver: bool = True,
    visual_output_format: str = "pvd",
    gradient_export_mode: str = "article_safe",
    export_material_fields: bool = True,
    export_mesh: bool = True,
    export_sampling_conditions: bool = True,
    export_initial_state: bool = False,
    export_final_state: bool = True,
    article_ready: bool = False,
    common_mesh_direct_pair: bool = True,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    if every_k <= 0:
        raise ValueError("every_k must be positive.")

    visual_output_format = str(visual_output_format).lower()
    if visual_output_format not in {"pvd", "xdmf", "both"}:
        raise ValueError("visual_output_format must be one of 'pvd', 'xdmf', or 'both'.")
    export_xdmf = visual_output_format in {"xdmf", "both"}
    export_pvd = True

    gradient_export_specs = _gradient_export_specs(gradient_export_mode)
    primary_gradient_spec = gradient_export_specs[0]

    observation_kind = str(observation_kind)
    allowed_observation_kinds = {"scalar_energy", "vector_array"}
    if observation_kind not in allowed_observation_kinds:
        raise ValueError(
            f"observation_kind must be one of {sorted(allowed_observation_kinds)}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # Effective receiver line coordinates used only by DirectObservationPipelineConfig.
    #
    # DirectObservationPipelineConfig validates an
    # internal_gaussian_array receiver through line-array coordinates.
    # For receiver_array_geometry="box_except_gamma1", the true Gamma2
    # geometry is built later as an explicit polyline left+top+right.
    effective_receiver_array_start_x = receiver_array_start_x
    effective_receiver_array_start_y = receiver_array_start_y
    effective_receiver_array_end_x = receiver_array_end_x
    effective_receiver_array_end_y = receiver_array_end_y

    if (
        observation_kind == "vector_array"
        and str(receiver_array_geometry) == "box_except_gamma1"
    ):
        required_box_values = {
            "receiver_box_xmin": receiver_box_xmin,
            "receiver_box_xmax": receiver_box_xmax,
            "receiver_box_ymin": receiver_box_ymin,
            "receiver_box_ymax": receiver_box_ymax,
        }
        missing_box_values = [
            name for name, value in required_box_values.items()
            if value is None
        ]
        if missing_box_values:
            raise ValueError(
                "receiver_array_geometry='box_except_gamma1' requires "
                + ", ".join(missing_box_values)
                + "."
            )

        if effective_receiver_array_start_x is None:
            effective_receiver_array_start_x = float(receiver_box_xmin)
        if effective_receiver_array_start_y is None:
            effective_receiver_array_start_y = float(receiver_box_ymax)
        if effective_receiver_array_end_x is None:
            effective_receiver_array_end_x = float(receiver_box_xmax)
        if effective_receiver_array_end_y is None:
            effective_receiver_array_end_y = float(receiver_box_ymax)

    common_kwargs = dict(
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
        source_side=source_side,
        source_center=source_center,
        source_length=source_length,
        source_array_start=source_array_start,
        source_array_end=source_array_end,
        source_array_num_elements=source_array_num_elements,
        source_array_element_length=source_array_element_length,
        source_center_x=source_center_x,
        source_center_y=source_center_y,
        source_radius=source_radius,
        source_sigma=source_sigma,
        source_direction_x=source_direction_x,
        source_direction_y=source_direction_y,
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
        receiver_array_num_elements=receiver_array_num_elements,
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
        source_num_periods=source_num_periods,
        ksp_type=ksp_type,
        pc_type=pc_type,
        ksp_rtol=ksp_rtol,
        ksp_atol=ksp_atol,
        ksp_max_it=ksp_max_it,
        store_displacement_snapshots=True,
        reuse_effective_solver=bool(reuse_direct_effective_solver),
    )

    # Compatibility bridge for vector_array Gamma2 box geometry.
    #
    # DirectObservationPipelineConfig still validates receiver_kind="internal_gaussian_array"
    # through the line-array coordinates:
    # receiver_array_start_x/y and receiver_array_end_x/y.
    #
    # In vector_array mode receiver_array_geometry="box_except_gamma1",
    # the actual Gamma2 geometry is built later from receiver_box_* as a polyline
    # left + top + right.  The following fallback only provides a valid single
    # top-line receiver to satisfy the direct configuration validation.  It does
    # not replace the Gamma2 polyline used by the vector_array observation.
    if (
        observation_kind == "vector_array"
        and str(receiver_array_geometry) == "box_except_gamma1"
    ):
        required_box_values = {
            "receiver_box_xmin": receiver_box_xmin,
            "receiver_box_xmax": receiver_box_xmax,
            "receiver_box_ymin": receiver_box_ymin,
            "receiver_box_ymax": receiver_box_ymax,
        }
        missing_box_values = [
            name for name, value in required_box_values.items()
            if value is None
        ]
        if missing_box_values:
            raise ValueError(
                "receiver_array_geometry='box_except_gamma1' requires "
                + ", ".join(missing_box_values)
                + "."
            )

        if receiver_array_start_x is None:
            receiver_array_start_x = float(receiver_box_xmin)
        if receiver_array_start_y is None:
            receiver_array_start_y = float(receiver_box_ymax)
        if receiver_array_end_x is None:
            receiver_array_end_x = float(receiver_box_xmax)
        if receiver_array_end_y is None:
            receiver_array_end_y = float(receiver_box_ymax)

    # Effective receiver line coordinates used only by DirectObservationPipelineConfig.
    #
    # For receiver_array_geometry="box_except_gamma1", the real Gamma2 geometry is
    # the polyline left+top+right built later by vector_array.  However, the direct
    # forward pipeline still validates receiver_kind="internal_gaussian_array" as
    # a single line receiver.  These effective coordinates provide that
    # compatibility line on the top side of the observation box.
    effective_receiver_array_start_x = receiver_array_start_x
    effective_receiver_array_start_y = receiver_array_start_y
    effective_receiver_array_end_x = receiver_array_end_x
    effective_receiver_array_end_y = receiver_array_end_y

    if (
        observation_kind == "vector_array"
        and str(receiver_array_geometry) == "box_except_gamma1"
    ):
        required_box_values = {
            "receiver_box_xmin": receiver_box_xmin,
            "receiver_box_xmax": receiver_box_xmax,
            "receiver_box_ymin": receiver_box_ymin,
            "receiver_box_ymax": receiver_box_ymax,
        }
        missing_box_values = [
            name for name, value in required_box_values.items()
            if value is None
        ]
        if missing_box_values:
            raise ValueError(
                "receiver_array_geometry='box_except_gamma1' requires "
                + ", ".join(missing_box_values)
                + "."
            )

        if effective_receiver_array_start_x is None:
            effective_receiver_array_start_x = float(receiver_box_xmin)
        if effective_receiver_array_start_y is None:
            effective_receiver_array_start_y = float(receiver_box_ymax)
        if effective_receiver_array_end_x is None:
            effective_receiver_array_end_x = float(receiver_box_xmax)
        if effective_receiver_array_end_y is None:
            effective_receiver_array_end_y = float(receiver_box_ymax)

    simulated_config = _build_direct_config(
        **common_kwargs,
        material_kind=simulated_material_kind,
    )

    observed_config = _build_direct_config(
        **common_kwargs,
        material_kind=observed_material_kind,
    )

    comparison = None
    vector_comparison = None
    adjoint_rhs = None
    adjoint = None
    matrices = None

    reflection_single_source_rhs = None
    transmission_single_source_rhs = None
    reflection_residual_wave = None
    transmission_residual_wave = None
    residual_wave_exports = None
    residual_wave_rhs_manifest = None
    residual_wave_solve_manifest = None

    try:
        comparison = run_direct_observation_comparison(
            DirectObservationComparisonConfig(
                simulated=simulated_config,
                observed=observed_config,
                weights=ObservationFunctionalWeights(
                    reflection=reflection_weight,
                    transmission=transmission_weight,
                ),
                mesh_policy=(
                    "shared_observed_refined"
                    if common_mesh_direct_pair
                    else "independent"
                ),
            ),
            comm=comm,
        )

        V = comparison.simulated.function_space
        if V is None:
            raise RuntimeError("Simulated direct result does not expose function_space.")

        if common_mesh_direct_pair:
            common_mesh = comparison.common_mesh_metadata
            required_common_mesh_checks = {
                "common_mesh": common_mesh.get("common_mesh") is True,
                "mesh_created_once": common_mesh.get("mesh_created_once") is True,
                "shared_mesh_object": common_mesh.get("shared_mesh_object") is True,
                "shared_function_space": common_mesh.get("shared_function_space") is True,
                "same_mpi_partition": common_mesh.get("same_mpi_partition") is True,
                "same_export_times": common_mesh.get("same_export_times") is True,
            }
            failed = [
                name for name, ok in required_common_mesh_checks.items() if not ok
            ]
            if failed:
                raise RuntimeError(
                    "Common-mesh direct contract failed before adjoint/imaging "
                    "solves: " + ", ".join(failed)
                )
            if comparison.observed.function_space is not V:
                raise RuntimeError(
                    "Common-mesh direct contract requires one shared finite-"
                    "element space for healthy and defective direct solves."
                )

        boundary_specs, _ = _build_source_receiver_boundary_specs(simulated_config)
        facet_tags = build_boundary_meshtags(V.mesh, boundary_specs)

        materials, material_metadata = _build_materials(V.mesh, simulated_config)
        absorbing_sigma, absorbing_layer_metadata = _build_absorbing_layer_if_requested(
            V.mesh,
            simulated_config,
        )

        matrices = assemble_elastic_operator_matrices(
            V,
            materials,
            ElasticOperatorConfig(theta=theta, dt=dt),
            damping_coefficient=absorbing_sigma,
        )

        vector_array_artifacts: dict[str, Any] | None = None
        resolved_receiver_sigma: float | None = None

        if observation_kind == "vector_array":
            (
                reflection_vector_spec,
                transmission_vector_spec,
                resolved_receiver_sigma,
            ) = _build_vector_array_observation_specs_from_config(
                config=simulated_config,
                receiver_sigma=receiver_sigma,
                receiver_array_geometry=receiver_array_geometry,
                receiver_box_xmin=receiver_box_xmin,
                receiver_box_xmax=receiver_box_xmax,
                receiver_box_ymin=receiver_box_ymin,
                receiver_box_ymax=receiver_box_ymax,
                receiver_box_left_num_elements=receiver_box_left_num_elements,
                receiver_box_top_num_elements=receiver_box_top_num_elements,
                receiver_box_right_num_elements=receiver_box_right_num_elements,
            )

            vector_comparison = run_vector_array_observation_comparison_from_direct_results(
                simulated=comparison.simulated,
                observed=comparison.observed,
                reflection_spec=reflection_vector_spec,
                transmission_spec=transmission_vector_spec,
                reflection_scale=reflection_weight,
                transmission_scale=transmission_weight,
            )

            adjoint_rhs = vector_comparison.adjoint_rhs

        else:
            internal_transmission_spec = None
            if simulated_config.receiver_kind == "internal_gaussian_array":
                internal_receiver_sigma = (
                    simulated_config.receiver_array_element_length
                    if simulated_config.receiver_array_element_length is not None
                    else simulated_config.source_sigma
                )
                internal_transmission_spec = internal_gaussian_array_observation(
                    start_x=simulated_config.receiver_array_start_x,
                    start_y=simulated_config.receiver_array_start_y,
                    end_x=simulated_config.receiver_array_end_x,
                    end_y=simulated_config.receiver_array_end_y,
                    num_elements=simulated_config.receiver_array_num_elements,
                    sigma=internal_receiver_sigma,
                    component="vector",
                    tag=2,
                )

            adjoint_rhs = assemble_full_adjoint_observation_rhs(
                V,
                facet_tags,
                comparison.simulated.displacement_snapshots,
                comparison.residuals,
                reflection_scale=reflection_weight,
                transmission_scale=transmission_weight,
                internal_transmission_spec=internal_transmission_spec,
            )

        adjoint = solve_newmark_discrete_adjoint_state(
            matrices,
            adjoint_rhs,
            solve_config=LinearSolveConfig(
                ksp_type=ksp_type,
                pc_type=pc_type,
                rtol=ksp_rtol,
                atol=ksp_atol,
                max_it=ksp_max_it,
            ),
            reuse_effective_solver=bool(reuse_adjoint_effective_solver),
        )

        direct_sim_export = _export_vector_time_series_xdmf(
            output_path=output_dir / "direct_simulated_displacement.xdmf",
            V=V,
            snapshots=comparison.simulated.displacement_snapshots,
            dt=dt,
            every_k=every_k,
            field_name="direct_simulated_displacement",
            comm=comm,
            export_xdmf=export_xdmf,
            time_records=comparison.simulated.time_loop_records,
            export_initial_state=export_initial_state,
            export_final_state=export_final_state,
        )

        V_observed = comparison.observed.function_space
        if V_observed is None:
            raise RuntimeError("Observed direct result does not expose function_space.")

        direct_obs_export = _export_vector_time_series_xdmf(
            output_path=output_dir / "direct_observed_displacement.xdmf",
            V=V_observed,
            snapshots=comparison.observed.displacement_snapshots,
            dt=dt,
            every_k=every_k,
            field_name="direct_observed_displacement",
            comm=comm,
            export_xdmf=export_xdmf,
            time_records=comparison.observed.time_loop_records,
            export_initial_state=export_initial_state,
            export_final_state=export_final_state,
        )

        adjoint_export = _export_vector_time_series_xdmf(
            output_path=output_dir / "adjoint_displacement.xdmf",
            V=V,
            snapshots=adjoint.equation_adjoint_vectors,
            dt=dt,
            every_k=every_k,
            field_name="adjoint_displacement",
            comm=comm,
            export_xdmf=export_xdmf,
            time_records=adjoint.records,
            export_initial_state=export_initial_state,
            export_final_state=export_final_state,
            initial_snapshot=adjoint.initial_current_adjoint,
        )

        adjoint_solve_order_export = _export_vector_time_series_xdmf(
            output_path=output_dir / "adjoint_solve_order_displacement.xdmf",
            V=V,
            snapshots=tuple(reversed(adjoint.equation_adjoint_vectors)),
            dt=dt,
            every_k=every_k,
            field_name="adjoint_solve_order_displacement",
            comm=comm,
            export_xdmf=export_xdmf,
            time_records=tuple(reversed(adjoint.records)),
            export_initial_state=False,
            export_final_state=export_final_state,
        )

        if observation_kind == "vector_array":
            reflection_single_source_rhs = make_single_source_vector_array_full_rhs(
                adjoint_rhs,
                keep="reflection",
            )
            transmission_single_source_rhs = make_single_source_vector_array_full_rhs(
                adjoint_rhs,
                keep="transmission",
            )

            reflection_residual_wave = solve_newmark_discrete_adjoint_state(
                matrices,
                reflection_single_source_rhs.full_rhs,
                solve_config=LinearSolveConfig(
                    ksp_type=ksp_type,
                    pc_type=pc_type,
                    rtol=ksp_rtol,
                    atol=ksp_atol,
                    max_it=ksp_max_it,
                ),
                reuse_effective_solver=bool(reuse_adjoint_effective_solver),
            )

            transmission_residual_wave = solve_newmark_discrete_adjoint_state(
                matrices,
                transmission_single_source_rhs.full_rhs,
                solve_config=LinearSolveConfig(
                    ksp_type=ksp_type,
                    pc_type=pc_type,
                    rtol=ksp_rtol,
                    atol=ksp_atol,
                    max_it=ksp_max_it,
                ),
                reuse_effective_solver=bool(reuse_adjoint_effective_solver),
            )

            reflection_residual_wave_export = _export_vector_time_series_xdmf(
                output_path=output_dir / "reflection_residual_wavefield.xdmf",
                V=V,
                snapshots=reflection_residual_wave.equation_adjoint_vectors,
                dt=dt,
                every_k=every_k,
                field_name="reflection_residual_wavefield",
                comm=comm,
                export_xdmf=export_xdmf,
                time_records=reflection_residual_wave.records,
                export_initial_state=export_initial_state,
                export_final_state=export_final_state,
                initial_snapshot=reflection_residual_wave.initial_current_adjoint,
            )

            transmission_residual_wave_export = _export_vector_time_series_xdmf(
                output_path=output_dir / "transmission_residual_wavefield.xdmf",
                V=V,
                snapshots=transmission_residual_wave.equation_adjoint_vectors,
                dt=dt,
                every_k=every_k,
                field_name="transmission_residual_wavefield",
                comm=comm,
                export_xdmf=export_xdmf,
                time_records=transmission_residual_wave.records,
                export_initial_state=export_initial_state,
                export_final_state=export_final_state,
                initial_snapshot=transmission_residual_wave.initial_current_adjoint,
            )

            residual_wave_material_parameter_gradients_export = (
                _export_residual_wave_material_parameter_gradients_xdmf(
                    output_dir=output_dir,
                    V=V,
                    reflection_residual_wave_snapshots=reflection_residual_wave.equation_adjoint_vectors,
                    transmission_residual_wave_snapshots=transmission_residual_wave.equation_adjoint_vectors,
                    dt=dt,
                    comm=comm,
            export_xdmf=export_xdmf,
                )
            )

            residual_wave_interaction_image_export = (
                _export_residual_wave_interaction_image_xdmf(
                    output_dir=output_dir,
                    V=V,
                    reflection_residual_wave_snapshots=reflection_residual_wave.equation_adjoint_vectors,
                    transmission_residual_wave_snapshots=transmission_residual_wave.equation_adjoint_vectors,
                    dt=dt,
                    comm=comm,
            export_xdmf=export_xdmf,
                )
            )

            residual_wave_rhs_manifest = {
                "kind": "residual_wave_split_rhs",
                "reflection": reflection_single_source_rhs.as_dict(),
                "transmission": transmission_single_source_rhs.as_dict(),
            }

            residual_wave_solve_manifest = {
                "kind": "residual_wave_split_solve",
                "reuse_effective_solver": bool(reuse_adjoint_effective_solver),
                "reflection": reflection_residual_wave.as_dict(),
                "transmission": transmission_residual_wave.as_dict(),
            }

            residual_wave_exports = {
                "status": "ok",
                "imaging_mode": "residual_wave_interaction",
                "reflection_residual_wavefield": reflection_residual_wave_export,
                "transmission_residual_wavefield": transmission_residual_wave_export,
                "material_parameter_kernels": residual_wave_material_parameter_gradients_export,
                "interaction_image": residual_wave_interaction_image_export,
                "paraview_entrypoints": [
                    reflection_residual_wave_export["paraview_entrypoint"],
                    transmission_residual_wave_export["paraview_entrypoint"],
                    residual_wave_material_parameter_gradients_export["paraview_entrypoint"],
                    residual_wave_interaction_image_export["paraview_entrypoint"],
                ],
            }

        gradient_export = _export_gradient_proxy_xdmf_and_npz(
            output_dir=output_dir,
            V=V,
            direct_snapshots=comparison.simulated.displacement_snapshots,
            adjoint_vectors=adjoint.equation_adjoint_vectors,
            comm=comm,
            export_xdmf=export_xdmf,
        )

        material_parameter_gradients_export = _export_material_parameter_gradients_xdmf(
            output_dir=output_dir,
            V=V,
            direct_snapshots=comparison.simulated.displacement_snapshots,
            adjoint_snapshots=adjoint.equation_adjoint_vectors,
            dt=dt,
            theta=theta,
            comm=comm,
            export_xdmf=export_xdmf,
            absolute_value=bool(primary_gradient_spec["absolute_value"]),
            normalize=bool(primary_gradient_spec["normalize"]),
            gradient_export_label=str(primary_gradient_spec["label"]),
        )

        interaction_image_export = _export_direct_adjoint_interaction_image_xdmf(
            output_dir=output_dir,
            V=V,
            direct_snapshots=comparison.simulated.displacement_snapshots,
            adjoint_snapshots=adjoint.equation_adjoint_vectors,
            dt=dt,
            theta=theta,
            comm=comm,
            export_xdmf=export_xdmf,
        )

        if observation_kind == "vector_array" and (
            reflection_residual_wave is None or transmission_residual_wave is None
        ):
            raise RuntimeError(
                "Internal error: vector_array gradient combinations require "
                "reflection and transmission residual wavefields."
            )

        gradient_combination_sets: dict[str, Any] = {}
        for spec in gradient_export_specs:
            gradient_combination_sets[str(spec["manifest_key"])] = (
                _export_gradient_combination_set(
                    output_dir=output_dir / str(spec["subdir"]),
                    V=V,
                    comparison=comparison,
                    adjoint=adjoint,
                    reflection_residual_wave=reflection_residual_wave,
                    transmission_residual_wave=transmission_residual_wave,
                    dt=dt,
                    theta=theta,
                    comm=comm,
                    export_xdmf=export_xdmf,
                    absolute_value=bool(spec["absolute_value"]),
                    normalize=bool(spec["normalize"]),
                    gradient_export_label=str(spec["label"]),
                )
            )

        gradient_combinations_export = gradient_combination_sets["gradient_combinations"]
        gradient_combinations_raw_signed_export = gradient_combination_sets.get(
            "gradient_combinations_raw_signed"
        )

        residual_export = _export_residuals_csv_png(
            output_dir=output_dir,
            residuals=comparison.residuals,
            comm=comm,
        )

        if observation_kind == "vector_array":
            if vector_comparison is None or resolved_receiver_sigma is None:
                raise RuntimeError("Internal error: missing vector_array comparison.")
            vector_array_artifacts = _export_vector_array_artifacts(
                output_dir=output_dir,
                vector_comparison=vector_comparison,
                receiver_sigma=resolved_receiver_sigma,
                comm=comm,
            )

        material_export: dict[str, Any] | None = None
        if export_material_fields:
            simulated_material_export = _export_material_fields_xdmf(
                output_dir=output_dir / "material_fields" / "simulated_healthy",
                config=simulated_config,
                msh=V.mesh,
                comm=comm,
                export_xdmf=export_xdmf,
            )
            observed_material_export = _export_material_fields_xdmf(
                output_dir=output_dir / "material_fields" / "observed_defective",
                config=observed_config,
                msh=V.mesh,
                comm=comm,
                export_xdmf=export_xdmf,
            )
            material_export = {
                "status": "ok",
                "simulated_healthy": simulated_material_export,
                "observed_defective": observed_material_export,
                "paraview_entrypoint": observed_material_export["paraview_entrypoint"],
                "paraview_entrypoint_xdmf": observed_material_export[
                    "paraview_entrypoint_xdmf"
                ],
            }

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
                config=simulated_config,
                comm=comm,
            )

        tdim = int(V.mesh.topology.dim)
        vertex_map = V.mesh.topology.index_map(0)
        cell_map = V.mesh.topology.index_map(tdim)
        dof_map = V.dofmap.index_map
        dof_block_size = int(V.dofmap.index_map_bs)
        mesh_partition_fingerprint = _mesh_partition_fingerprint(
            V.mesh,
            comm,
        )

        mesh_discretization = {
            "topological_dimension": tdim,
            "global_owned_vertices": int(
                comm.allreduce(int(vertex_map.size_local), op=MPI.SUM)
            ),
            "global_owned_cells": int(
                comm.allreduce(int(cell_map.size_local), op=MPI.SUM)
            ),
            "global_displacement_dofs": int(dof_map.size_global * dof_block_size),
            "displacement_dof_block_size": dof_block_size,
        }

        temporal_sampling = {
            "T_final": float(dt * num_steps),
            "dt": float(dt),
            "num_steps": int(num_steps),
            "receiver_sampling_stride": 1,
            "receiver_sampling_period": float(dt),
            "volume_sampling_stride": int(every_k),
            "volume_sampling_period": float(every_k * dt),
            "export_initial_state": bool(export_initial_state),
            "export_final_state": bool(export_final_state),
            "receiver_sampling_semantics": "every_computed_solver_step",
        }

        article_ready_exports = {
            "status": "ok",
            "raw_signed_gradients": gradient_combinations_raw_signed_export is not None,
            "article_safe_gradients": gradient_combinations_export is not None,
            "receiver_signals_npz": vector_array_artifacts is not None,
            "receiver_residuals_csv": vector_array_artifacts is not None,
            "receiver_coordinates_csv": vector_array_artifacts is not None,
            "residual_wavefields": residual_wave_exports is not None,
            "healthy_and_defective_material_fields": material_export is not None,
            "mesh_export": mesh_export is not None,
            "sampling_conditions": sampling_export is not None,
            "physical_receiver_times": True,
            "physical_volume_times": True,
            "common_mesh_direct_pair": bool(common_mesh_direct_pair),
            "all_imaging_wave_problems_use_shared_space": bool(
                common_mesh_direct_pair
            ),
        }

        manifest = {
            "kind": "inverse_time_series_xdmf",
            "status": "ok",
            "mpi_rank": comm.rank,
            "mpi_size": comm.size,
            "nx": int(nx),
            "ny": int(ny),
            "degree": int(degree),
            "domain_length_x": float(domain_length_x),
            "domain_length_y": float(domain_length_y),
            "theta": float(theta),
            "dt": float(dt),
            "num_steps": int(num_steps),
            "final_time": float(dt * num_steps),
            "every_k": int(every_k),
            "temporal_sampling": temporal_sampling,
            "mesh_discretization": mesh_discretization,
            "source_time_function": {
                "kind": "hann_windowed_sine_pulse",
                "amplitude": float(simulated_config.source_amplitude),
                "frequency": float(source_frequency),
                "num_periods": float(source_num_periods),
                "start_time": float(simulated_config.start_time),
                "duration": float(source_num_periods / source_frequency),
            },
            "article_ready": bool(article_ready),
            "common_mesh_direct_pair": bool(common_mesh_direct_pair),
            "direct_mesh_contract": {
                "status": (
                    "ok"
                    if (
                        not common_mesh_direct_pair
                        or comparison.common_mesh_metadata.get("common_mesh") is True
                    )
                    else "failed"
                ),
                "policy": (
                    "shared_observed_refined"
                    if common_mesh_direct_pair
                    else "independent"
                ),
                "mesh_source": comparison.common_mesh_metadata.get("mesh_source"),
                "refinement_semantics": comparison.common_mesh_metadata.get(
                    "refinement_semantics"
                ),
                "mesh_created_once": comparison.common_mesh_metadata.get(
                    "mesh_created_once", False
                ),
                "shared_mesh_object": comparison.common_mesh_metadata.get(
                    "shared_mesh_object", False
                ),
                "shared_function_space": comparison.common_mesh_metadata.get(
                    "shared_function_space", False
                ),
                "same_mpi_partition": comparison.common_mesh_metadata.get(
                    "same_mpi_partition", False
                ),
                "same_export_times": comparison.common_mesh_metadata.get(
                    "same_export_times", False
                ),
                "refinement_disks": comparison.common_mesh_metadata.get(
                    "refinement_disks", []
                ),
                "partition_fingerprint": mesh_partition_fingerprint,
                "all_imaging_wave_problems_use_shared_space": bool(
                    common_mesh_direct_pair
                ),
            },
            "article_ready_exports": article_ready_exports,
            "visual_output_format": visual_output_format,
            "export_pvd": bool(export_pvd),
            "export_xdmf": bool(export_xdmf),
            "simulated_material_kind": simulated_material_kind,
            "observed_material_kind": observed_material_kind,
            "defect_center_x": float(defect_center_x),
            "defect_center_y": float(defect_center_y),
            "defect_radius": float(defect_radius),
            "contrast_factor": float(contrast_factor),
            "reuse_direct_effective_solver": bool(reuse_direct_effective_solver),
            "reuse_adjoint_effective_solver": bool(reuse_adjoint_effective_solver),
            "observation_kind": observation_kind,
            "receiver_sigma": None if receiver_sigma is None else float(receiver_sigma),
            "resolved_receiver_sigma": (
                None
                if resolved_receiver_sigma is None
                else float(resolved_receiver_sigma)
            ),
            "source_kind": source_kind,
            "source_side": source_side,
            "source_array_start": source_array_start,
            "source_array_end": source_array_end,
            "source_array_num_elements": int(source_array_num_elements),
            "receiver_kind": receiver_kind,
            "receiver_side": receiver_side,
            "receiver_array_start": receiver_array_start,
            "receiver_array_end": receiver_array_end,
            "receiver_array_num_elements": int(receiver_array_num_elements),
            "receiver_array_geometry": str(receiver_array_geometry),
            "receiver_box": {
                "xmin": None if receiver_box_xmin is None else float(receiver_box_xmin),
                "xmax": None if receiver_box_xmax is None else float(receiver_box_xmax),
                "ymin": None if receiver_box_ymin is None else float(receiver_box_ymin),
                "ymax": None if receiver_box_ymax is None else float(receiver_box_ymax),
                "left_num_elements": int(receiver_box_left_num_elements),
                "top_num_elements": int(receiver_box_top_num_elements),
                "right_num_elements": int(receiver_box_right_num_elements),
            },
            "absorbing_layer": absorbing_layer_metadata,
            "material_metadata": material_metadata,
            "comparison": comparison.as_dict(),
            "vector_array_comparison": (
                None
                if vector_comparison is None
                else {
                    **vector_comparison.as_dict(),
                    "observation_specs": {
                        "reflection": reflection_vector_spec.as_dict(),
                        "transmission": transmission_vector_spec.as_dict(),
                    },
                }
            ),
            "adjoint_rhs": adjoint_rhs.as_dict(),
            "residual_wave_rhs": residual_wave_rhs_manifest,
            "residual_wave_solve": residual_wave_solve_manifest,
            "adjoint": adjoint.as_dict(),
            "exports": {
                "direct_simulated": direct_sim_export,
                "direct_observed": direct_obs_export,
                "adjoint": adjoint_export,
                "adjoint_solve_order": adjoint_solve_order_export,
                "gradient_proxy": gradient_export,
                "material_parameter_gradients": material_parameter_gradients_export,
                "direct_adjoint_interaction_image": interaction_image_export,
                "gradient_combinations": gradient_combinations_export,
                "gradient_combinations_raw_signed": gradient_combinations_raw_signed_export,
                "gradient_export_mode": str(gradient_export_mode),
                "gradient_export_specs": gradient_export_specs,
                "residual_wave_interaction": residual_wave_exports,
                "residuals": residual_export,
                "vector_array": vector_array_artifacts,
                "material_fields": material_export,
                "mesh": mesh_export,
                "sampling_conditions": sampling_export,
            },
            "paraview_entrypoints": [
                direct_sim_export["paraview_entrypoint"],
                direct_obs_export["paraview_entrypoint"],
                adjoint_export["paraview_entrypoint"],
                adjoint_solve_order_export["paraview_entrypoint"],
                gradient_export["paraview_entrypoint"],
                material_parameter_gradients_export["paraview_entrypoint"],
                interaction_image_export["paraview_entrypoint"],
                *[
                    payload["paraview_entrypoint"]
                    for payload in gradient_combinations_export.values()
                    if isinstance(payload, dict) and payload.get("paraview_entrypoint")
                ],
                *([] if gradient_combinations_raw_signed_export is None else [
                    payload["paraview_entrypoint"]
                    for payload in gradient_combinations_raw_signed_export.values()
                    if isinstance(payload, dict) and payload.get("paraview_entrypoint")
                ]),
                (
                    None
                    if residual_wave_exports is None
                    else residual_wave_exports["interaction_image"]["paraview_entrypoint"]
                ),
                (
                    None
                    if residual_wave_exports is None
                    else residual_wave_exports["material_parameter_kernels"]["paraview_entrypoint"]
                ),
                None if material_export is None else material_export["paraview_entrypoint"],
                None if mesh_export is None else mesh_export["paraview_entrypoint"],
            ],
            "paraview_entrypoints_xdmf": [
                direct_sim_export["paraview_entrypoint_xdmf"],
                direct_obs_export["paraview_entrypoint_xdmf"],
                adjoint_export["paraview_entrypoint_xdmf"],
                adjoint_solve_order_export["paraview_entrypoint_xdmf"],
                gradient_export["paraview_entrypoint_xdmf"],
                material_parameter_gradients_export["paraview_entrypoint_xdmf"],
                interaction_image_export["paraview_entrypoint_xdmf"],
                *[
                    payload["paraview_entrypoint_xdmf"]
                    for payload in gradient_combinations_export.values()
                    if isinstance(payload, dict) and payload.get("paraview_entrypoint_xdmf")
                ],
                *([] if gradient_combinations_raw_signed_export is None else [
                    payload["paraview_entrypoint_xdmf"]
                    for payload in gradient_combinations_raw_signed_export.values()
                    if isinstance(payload, dict) and payload.get("paraview_entrypoint_xdmf")
                ]),
                (
                    None
                    if residual_wave_exports is None
                    else residual_wave_exports["interaction_image"]["paraview_entrypoint_xdmf"]
                ),
                (
                    None
                    if residual_wave_exports is None
                    else residual_wave_exports["material_parameter_kernels"]["paraview_entrypoint_xdmf"]
                ),
                None if material_export is None else material_export["paraview_entrypoint_xdmf"],
                None if mesh_export is None else mesh_export["paraview_entrypoint_xdmf"],
            ],
        }

        try:
            from elastic_fwi.tracking.tracked_manifest import enrich_manifest_with_tracking

            manifest = enrich_manifest_with_tracking(
                manifest,
                output_dir=output_dir,
                args={
                    "diagnostic": "export_inverse_time_series_xdmf",
                    "nx": int(nx),
                    "ny": int(ny),
                    "degree": int(degree),
                    "dt": float(dt),
                    "num_steps": int(num_steps),
                    "every_k": int(every_k),
                    "visual_output_format": visual_output_format,
                    "gradient_export_mode": str(gradient_export_mode),
                    "observation_kind": observation_kind,
                    "simulated_material_kind": simulated_material_kind,
                    "observed_material_kind": observed_material_kind,
                    "export_material_fields": bool(export_material_fields),
                    "export_mesh": bool(export_mesh),
                    "export_sampling_conditions": bool(export_sampling_conditions),
                    "export_initial_state": bool(export_initial_state),
                    "export_final_state": bool(export_final_state),
                    "article_ready": bool(article_ready),
                },
                comm=comm,
            )
        except Exception as exc:
            manifest["tracking_error"] = repr(exc)


        manifest_path = _ranked_path(output_dir / "inverse_time_series_manifest.json", comm)
        manifest_path.write_text(
            json.dumps(_to_jsonable(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    

    finally:
        if matrices is not None:
            matrices.destroy()

        if reflection_residual_wave is not None:
            reflection_residual_wave.destroy()

        if transmission_residual_wave is not None:
            transmission_residual_wave.destroy()

        if reflection_single_source_rhs is not None:
            reflection_single_source_rhs.destroy_zero_rhs()

        if transmission_single_source_rhs is not None:
            transmission_single_source_rhs.destroy_zero_rhs()

        if adjoint is not None:
            adjoint.destroy()

        if adjoint_rhs is not None:
            adjoint_rhs.destroy()

        if comparison is not None:
            comparison.simulated.destroy()
            comparison.observed.destroy()

    return manifest



def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
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
    parser.add_argument("--absorbing-layer-strength", type=float, default=3.0)
    parser.add_argument("--absorbing-layer-polynomial-order", type=float, default=2.0)
    parser.add_argument("--absorbing-layer-min-cells", type=int, default=6)
    parser.add_argument("--absorbing-layer-max-fraction", type=float, default=0.45)

    parser.add_argument("--domain-length-x", type=float, default=1.0)
    parser.add_argument("--domain-length-y", type=float, default=1.0)

    parser.add_argument(
        "--observation-kind",
        type=str,
        default="scalar_energy",
        choices=("scalar_energy", "vector_array"),
        help=(
            "Observation model used to build the adjoint RHS. "
            "scalar_energy keeps the historical diagnostic branch; "
            "vector_array uses ux,uy transducer-array residuals on Gamma1/Gamma2."
        ),
    )

    parser.add_argument("--source-kind", type=str, default="boundary_segment")
    parser.add_argument("--source-side", type=str, default="bottom", choices=("bottom", "top", "left", "right"))
    parser.add_argument("--source-center", type=float, default=None)
    parser.add_argument("--source-length", type=float, default=None)
    parser.add_argument("--source-array-start", type=float, default=None)
    parser.add_argument("--source-array-end", type=float, default=None)
    parser.add_argument("--source-array-num-elements", type=int, default=1)
    parser.add_argument("--source-array-element-length", type=float, default=None)
    parser.add_argument("--source-center-x", type=float, default=0.5)
    parser.add_argument("--source-center-y", type=float, default=0.5)
    parser.add_argument("--source-radius", type=float, default=0.05)
    parser.add_argument("--source-sigma", type=float, default=0.05)
    parser.add_argument("--source-direction-x", type=float, default=0.0)
    parser.add_argument("--source-direction-y", type=float, default=1.0)
    parser.add_argument("--source-array-start-x", type=float, default=None)
    parser.add_argument("--source-array-start-y", type=float, default=None)
    parser.add_argument("--source-array-end-x", type=float, default=None)
    parser.add_argument("--source-array-end-y", type=float, default=None)

    parser.add_argument("--receiver-kind", type=str, default="boundary_segment")
    parser.add_argument("--receiver-side", type=str, default="top", choices=("bottom", "top", "left", "right"))
    parser.add_argument("--receiver-center", type=float, default=None)
    parser.add_argument("--receiver-length", type=float, default=None)
    parser.add_argument("--receiver-array-start", type=float, default=None)
    parser.add_argument("--receiver-array-end", type=float, default=None)
    parser.add_argument("--receiver-array-num-elements", type=int, default=1)
    parser.add_argument("--receiver-array-element-length", type=float, default=None)
    parser.add_argument(
        "--receiver-sigma",
        type=float,
        default=None,
        help=(
            "Gaussian width sigma_rec for vector_array observations. "
            "If omitted, receiver-array-element-length is used as compatibility alias."
        ),
    )
    parser.add_argument("--receiver-array-start-x", type=float, default=None)
    parser.add_argument("--receiver-array-start-y", type=float, default=None)
    parser.add_argument("--receiver-array-end-x", type=float, default=None)
    parser.add_argument("--receiver-array-end-y", type=float, default=None)
    parser.add_argument(
        "--receiver-array-geometry",
        type=str,
        default="line",
        choices=("line", "box_except_gamma1"),
        help=(
            "Geometry used for Gamma2 in vector_array mode. "
            "line keeps the historical single receiver line; "
            "box_except_gamma1 uses left+top+right sides of an observation box."
        ),
    )
    parser.add_argument("--receiver-box-xmin", type=float, default=None)
    parser.add_argument("--receiver-box-xmax", type=float, default=None)
    parser.add_argument("--receiver-box-ymin", type=float, default=None)
    parser.add_argument("--receiver-box-ymax", type=float, default=None)
    parser.add_argument("--receiver-box-left-num-elements", type=int, default=13)
    parser.add_argument("--receiver-box-top-num-elements", type=int, default=11)
    parser.add_argument("--receiver-box-right-num-elements", type=int, default=13)
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

    parser.add_argument("--simulated-material-kind", type=str, default="healthy")
    parser.add_argument("--observed-material-kind", type=str, default="localized_disk_defect")
    parser.add_argument("--defect-center-x", type=float, default=0.5)
    parser.add_argument("--defect-center-y", type=float, default=0.5)
    parser.add_argument("--defect-radius", type=float, default=0.35)
    parser.add_argument("--contrast-factor", type=float, default=0.8)
    parser.add_argument("--defects-json", type=str, default=None)
    parser.add_argument("--defect-background-factor", type=float, default=1.0)
    parser.add_argument("--defect-conflict-policy", type=str, default="last", choices=("first", "last", "error"))

    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--source-frequency", type=float, default=1.0)
    parser.add_argument("--source-num-periods", type=float, default=1.0)

    parser.add_argument("--reflection-weight", type=float, default=1.0)
    parser.add_argument("--transmission-weight", type=float, default=1.0)
    parser.add_argument("--every-k", type=int, default=1)
    parser.add_argument(
        "--visual-output-format",
        type=str,
        default="pvd",
        choices=("pvd", "xdmf", "both"),
        help="Image/time-series output format. Default 'pvd' writes PVD/VTK only; 'both' keeps legacy XDMF/HDF5 sidecars.",
    )

    parser.add_argument("--ksp-type", type=str, default="cg")
    parser.add_argument("--pc-type", type=str, default="jacobi")
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--ksp-atol", type=float, default=1.0e-12)
    parser.add_argument("--ksp-max-it", type=int, default=1000)
    parser.add_argument(
        "--no-reuse-direct-effective-solver",
        action="store_true",
        help="Use the historical one-shot KSP path for direct time-loop solves.",
    )
    parser.add_argument(
        "--no-reuse-adjoint-effective-solver",
        action="store_true",
        help="Use the historical one-shot KSP path for adjoint/residual-wave solves.",
    )
    parser.add_argument(
        "--independent-direct-meshes",
        action="store_true",
        help=(
            "Legacy compatibility only: build simulated and observed direct "
            "meshes independently. The default is one shared mesh built from "
            "the observed/defective physical-experiment geometry."
        ),
    )

    parser.add_argument(
        "--gradient-export-mode",
        choices=("article_safe", "raw_signed", "both"),
        default="article_safe",
        help=(
            "Material-gradient export mode: article_safe exports normalized "
            "absolute-value maps; raw_signed exports signed non-normalized "
            "gradient densities; both exports both families."
        ),
    )

    parser.add_argument("--no-export-material-fields", action="store_true")
    parser.add_argument("--no-export-mesh", action="store_true")
    parser.add_argument("--no-export-sampling-conditions", action="store_true")
    parser.add_argument(
        "--no-vtk",
        action="store_true",
        help="Accepted for CLI compatibility. VTK/PVD export is enabled by default in this diagnostic.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export direct, observed, adjoint and gradient proxy time series for the inverse elastic FWI workflow."
    )
    _add_common_arguments(parser)
    args = parser.parse_args(argv)

    manifest = export_inverse_time_series_xdmf(
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
        observation_kind=args.observation_kind,
        source_kind=args.source_kind,
        source_side=args.source_side,
        source_center=args.source_center,
        source_length=args.source_length,
        source_array_start=args.source_array_start,
        source_array_end=args.source_array_end,
        source_array_num_elements=args.source_array_num_elements,
        source_array_element_length=args.source_array_element_length,
        source_center_x=args.source_center_x,
        source_center_y=args.source_center_y,
        source_radius=args.source_radius,
        source_sigma=args.source_sigma,
        source_direction_x=args.source_direction_x,
        source_direction_y=args.source_direction_y,
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
        receiver_sigma=args.receiver_sigma,
        receiver_array_start_x=args.receiver_array_start_x,
        receiver_array_start_y=args.receiver_array_start_y,
        receiver_array_end_x=args.receiver_array_end_x,
        receiver_array_end_y=args.receiver_array_end_y,
        receiver_array_geometry=args.receiver_array_geometry,
        receiver_box_xmin=args.receiver_box_xmin,
        receiver_box_xmax=args.receiver_box_xmax,
        receiver_box_ymin=args.receiver_box_ymin,
        receiver_box_ymax=args.receiver_box_ymax,
        receiver_box_left_num_elements=args.receiver_box_left_num_elements,
        receiver_box_top_num_elements=args.receiver_box_top_num_elements,
        receiver_box_right_num_elements=args.receiver_box_right_num_elements,
        auto_place_probes_inside_absorbing_layer=args.auto_place_probes_inside_absorbing_layer,
        source_segment_xmin=args.source_segment_xmin,
        source_segment_xmax=args.source_segment_xmax,
        receiver_segment_xmin=args.receiver_segment_xmin,
        receiver_segment_xmax=args.receiver_segment_xmax,
        simulated_material_kind=args.simulated_material_kind,
        observed_material_kind=args.observed_material_kind,
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
        source_num_periods=args.source_num_periods,
        reflection_weight=args.reflection_weight,
        transmission_weight=args.transmission_weight,
        every_k=args.every_k,
        visual_output_format=args.visual_output_format,
        gradient_export_mode=args.gradient_export_mode,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_max_it=args.ksp_max_it,
        reuse_direct_effective_solver=not args.no_reuse_direct_effective_solver,
        reuse_adjoint_effective_solver=not args.no_reuse_adjoint_effective_solver,
        common_mesh_direct_pair=not args.independent_direct_meshes,
        export_material_fields=not args.no_export_material_fields,
        export_mesh=not args.no_export_mesh,
        export_sampling_conditions=not args.no_export_sampling_conditions,
        comm=MPI.COMM_WORLD,
    )

    if MPI.COMM_WORLD.rank == 0:
        exports = manifest["exports"]
        print("[INVERSE TIME SERIES XDMF] OK")
        print("direct_simulated =", exports["direct_simulated"]["xdmf"])
        print("direct_observed  =", exports["direct_observed"]["xdmf"])
        print("adjoint          =", exports["adjoint"]["xdmf"])
        print("adjoint solve ord=", exports["adjoint_solve_order"]["xdmf"])
        print("gradient_proxy   =", exports["gradient_proxy"]["xdmf"])
        print("gradient_proxy_vtk=", exports["gradient_proxy"]["paraview_entrypoint"])
        print("material_grads   =", exports["material_parameter_gradients"]["xdmf"])
        print("material_grads_vtk=", exports["material_parameter_gradients"]["paraview_entrypoint"])
        print("interaction_img  =", exports["direct_adjoint_interaction_image"]["xdmf"])
        print("interaction_vtk  =", exports["direct_adjoint_interaction_image"]["paraview_entrypoint"])
        print("max residual     =", manifest["comparison"]["values"]["max_abs_residual"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
