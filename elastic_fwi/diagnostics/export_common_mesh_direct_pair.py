from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.diagnostics.export_direct_time_series_xdmf import (
    _export_vector_time_series_vtk,
)
from elastic_fwi.io.manifests import read_manifest, write_manifest
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    build_direct_observation_spatial_context,
    run_direct_observation_pipeline,
    validate_direct_observation_common_mesh_compatibility,
)


_CONFIG_FIELDS = {
    item.name for item in fields(DirectObservationPipelineConfig)
}


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _extract_pair_config_dicts(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    comparison = _mapping(manifest.get("comparison"))

    if comparison is not None:
        comparison_config = _mapping(comparison.get("config"))

        if comparison_config is not None:
            simulated = _mapping(comparison_config.get("simulated"))
            observed = _mapping(comparison_config.get("observed"))

            if simulated is not None and observed is not None:
                return simulated, observed

        simulated_result = _mapping(comparison.get("simulated"))
        observed_result = _mapping(comparison.get("observed"))

        if simulated_result is not None and observed_result is not None:
            simulated = _mapping(simulated_result.get("config"))
            observed = _mapping(observed_result.get("config"))

            if simulated is not None and observed is not None:
                return simulated, observed

    return manifest, manifest


def _build_config(
    raw: Mapping[str, Any],
    *,
    material_kind: str,
    fallback_manifest: Mapping[str, Any],
    num_steps_override: int | None,
) -> DirectObservationPipelineConfig:
    kwargs: dict[str, Any] = {}

    for name in _CONFIG_FIELDS:
        if name in raw:
            kwargs[name] = raw[name]
        elif name in fallback_manifest:
            kwargs[name] = fallback_manifest[name]

    if (
        "reuse_effective_solver" not in kwargs
        and "reuse_direct_effective_solver" in fallback_manifest
    ):
        kwargs["reuse_effective_solver"] = bool(
            fallback_manifest["reuse_direct_effective_solver"]
        )

    kwargs["material_kind"] = str(material_kind)
    kwargs["store_displacement_snapshots"] = True

    if num_steps_override is not None:
        if int(num_steps_override) <= 0:
            raise ValueError("num_steps_override must be strictly positive.")
        kwargs["num_steps"] = int(num_steps_override)

    return DirectObservationPipelineConfig(**kwargs)


def load_common_mesh_pair_configs(
    template_manifest: str | Path,
    *,
    num_steps_override: int | None = None,
) -> tuple[
    dict[str, Any],
    DirectObservationPipelineConfig,
    DirectObservationPipelineConfig,
]:
    template_manifest = Path(template_manifest)
    payload = read_manifest(template_manifest)

    simulated_raw, observed_raw = _extract_pair_config_dicts(payload)

    simulated_material_kind = str(
        simulated_raw.get(
            "material_kind",
            payload.get("simulated_material_kind", "healthy"),
        )
    )
    observed_material_kind = str(
        observed_raw.get(
            "material_kind",
            payload.get(
                "observed_material_kind",
                "localized_disk_defect",
            ),
        )
    )

    simulated = _build_config(
        simulated_raw,
        material_kind=simulated_material_kind,
        fallback_manifest=payload,
        num_steps_override=num_steps_override,
    )
    observed = _build_config(
        observed_raw,
        material_kind=observed_material_kind,
        fallback_manifest=payload,
        num_steps_override=num_steps_override,
    )

    validate_direct_observation_common_mesh_compatibility(
        observed,
        simulated,
    )

    return payload, simulated, observed


def _prepare_empty_output_directory(
    output_dir: Path,
    comm: MPI.Comm,
) -> None:
    error: str | None = None

    if comm.rank == 0:
        if output_dir.exists() and any(output_dir.iterdir()):
            error = f"Output directory is not empty: {output_dir}"
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "vtk").mkdir(parents=True, exist_ok=True)
            (output_dir / "manifests").mkdir(
                parents=True,
                exist_ok=True,
            )
            (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    error = comm.bcast(error, root=0)

    if error is not None:
        raise FileExistsError(error)

    comm.Barrier()


def _hash_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(
        np.asarray(values.shape, dtype=np.int64).tobytes()
    )
    digest.update(values.view(np.uint8).tobytes())
    return digest.hexdigest()


def _mesh_fingerprint(msh, comm: MPI.Comm) -> dict[str, Any]:
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
        "coordinates_sha256": _hash_array(
            np.asarray(msh.geometry.x)
        ),
        "connectivity_array_sha256": _hash_array(
            np.asarray(connectivity.array, dtype=np.int64)
        ),
        "connectivity_offsets_sha256": _hash_array(
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


def _template_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_export_records(
    *,
    num_snapshots: int,
    dt: float,
    start_time: float,
) -> list[dict[str, Any]]:
    return [
        {
            "step": int(index),
            "time": float(start_time + index * dt),
        }
        for index in range(num_snapshots)
    ]


def export_common_mesh_direct_pair(
    *,
    template_manifest: str | Path,
    output_dir: str | Path,
    every_k: int | None = None,
    num_steps_override: int | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    local_template_manifest = Path(template_manifest).resolve()
    local_output_dir = Path(output_dir).resolve()

    template_manifest = Path(
        comm.bcast(
            (
                str(local_template_manifest)
                if comm.rank == 0
                else None
            ),
            root=0,
        )
    )

    output_dir = Path(
        comm.bcast(
            (
                str(local_output_dir)
                if comm.rank == 0
                else None
            ),
            root=0,
        )
    )

    if every_k is not None and int(every_k) <= 0:
        raise ValueError("every_k must be strictly positive.")

    _prepare_empty_output_directory(output_dir, comm)

    (
        template_payload,
        simulated_config,
        observed_config,
    ) = load_common_mesh_pair_configs(
        template_manifest,
        num_steps_override=num_steps_override,
    )

    resolved_every_k = int(
        template_payload.get("every_k", 1)
        if every_k is None
        else every_k
    )

    if resolved_every_k <= 0:
        raise ValueError("Resolved every_k must be strictly positive.")

    started = time.perf_counter()

    spatial_context = build_direct_observation_spatial_context(
        observed_config,
        comm=comm,
    )

    mesh_fingerprint = _mesh_fingerprint(
        spatial_context.mesh,
        comm,
    )

    simulated_result = None
    observed_result = None

    simulated_export: dict[str, Any] | None = None
    observed_export: dict[str, Any] | None = None

    simulated_summary: dict[str, Any] | None = None
    observed_summary: dict[str, Any] | None = None

    timings: list[dict[str, Any]] = []

    try:
        t0 = time.perf_counter()

        simulated_result = run_direct_observation_pipeline(
            simulated_config,
            comm=comm,
            spatial_context=spatial_context,
        )

        timings.append(
            {
                "section": "simulated_direct_solve",
                "elapsed_seconds": float(time.perf_counter() - t0),
                "status": "ok",
            }
        )

        if (
            simulated_result.function_space
            is not spatial_context.function_space
        ):
            raise RuntimeError(
                "Simulated solve did not retain the shared function space."
            )

        simulated_records = _historical_export_records(
            num_snapshots=(
                simulated_result.num_displacement_snapshots
            ),
            dt=simulated_config.dt,
            start_time=simulated_config.start_time,
        )

        t0 = time.perf_counter()

        simulated_export = _export_vector_time_series_vtk(
            output_path=(
                output_dir
                / "vtk"
                / "direct_simulated_displacement.pvd"
            ),
            V=spatial_context.function_space,
            snapshots=simulated_result.displacement_snapshots,
            time_loop_records=simulated_records,
            every_k=resolved_every_k,
            field_name="direct_simulated_displacement",
            comm=comm,
        )

        timings.append(
            {
                "section": "simulated_vtk_export",
                "elapsed_seconds": float(time.perf_counter() - t0),
                "status": "ok",
            }
        )

        simulated_summary = simulated_result.as_dict()
        simulated_result.destroy()
        simulated_result = None

        comm.Barrier()

        t0 = time.perf_counter()

        observed_result = run_direct_observation_pipeline(
            observed_config,
            comm=comm,
            spatial_context=spatial_context,
        )

        timings.append(
            {
                "section": "observed_direct_solve",
                "elapsed_seconds": float(time.perf_counter() - t0),
                "status": "ok",
            }
        )

        if (
            observed_result.function_space
            is not spatial_context.function_space
        ):
            raise RuntimeError(
                "Observed solve did not retain the shared function space."
            )

        observed_records = _historical_export_records(
            num_snapshots=(
                observed_result.num_displacement_snapshots
            ),
            dt=observed_config.dt,
            start_time=observed_config.start_time,
        )

        t0 = time.perf_counter()

        observed_export = _export_vector_time_series_vtk(
            output_path=(
                output_dir
                / "vtk"
                / "direct_observed_displacement.pvd"
            ),
            V=spatial_context.function_space,
            snapshots=observed_result.displacement_snapshots,
            time_loop_records=observed_records,
            every_k=resolved_every_k,
            field_name="direct_observed_displacement",
            comm=comm,
        )

        timings.append(
            {
                "section": "observed_vtk_export",
                "elapsed_seconds": float(time.perf_counter() - t0),
                "status": "ok",
            }
        )

        observed_summary = observed_result.as_dict()
        observed_result.destroy()
        observed_result = None

        comm.Barrier()

        assert simulated_export is not None
        assert observed_export is not None
        assert simulated_summary is not None
        assert observed_summary is not None

        exported_times_match = bool(
            simulated_export["exported_times"]
            == observed_export["exported_times"]
        )
        exported_indices_match = bool(
            simulated_export["exported_indices"]
            == observed_export["exported_indices"]
        )

        pvd_file_checks: dict[str, bool] | None = None

        if comm.rank == 0:
            pvd_file_checks = {
                "simulated_pvd_exists": bool(
                    Path(simulated_export["pvd"]).is_file()
                ),
                "observed_pvd_exists": bool(
                    Path(observed_export["pvd"]).is_file()
                ),
            }

        pvd_file_checks = comm.bcast(
            pvd_file_checks,
            root=0,
        )

        assert pvd_file_checks is not None

        checks = {
            "common_mesh": True,
            "mesh_created_once": True,
            "shared_mesh_object": True,
            "shared_function_space": True,
            "same_mpi_partition": True,
            "same_export_times": exported_times_match,
            "same_export_indices": exported_indices_match,
            "two_direct_solves": True,
            "zero_adjoint_solves": True,
            **pvd_file_checks,
        }

        status = "ok" if all(checks.values()) else "failed"

        statuses = comm.allgather(status)

        if len(set(statuses)) != 1:
            raise RuntimeError(
                "Common-mesh export status differs between MPI ranks: "
                f"{statuses}"
            )

        manifest = {
            "kind": "common_mesh_direct_pair",
            "status": status,
            "mpi_rank": 0,
            "mpi_size": int(comm.size),
            "template_manifest": str(template_manifest),
            "template_manifest_sha256": _template_sha256(
                template_manifest
            ),
            "output_dir": str(output_dir),
            "common_mesh": True,
            "mesh_created_once": True,
            "shared_mesh_object": True,
            "shared_function_space": True,
            "same_mpi_partition": True,
            "mesh_source": "observed_configuration",
            "direct_solves": 2,
            "adjoint_solves": 0,
            "simulated_material_kind": (
                simulated_config.material_kind
            ),
            "observed_material_kind": observed_config.material_kind,
            "defect_center_x": float(
                observed_config.defect_center_x
            ),
            "defect_center_y": float(
                observed_config.defect_center_y
            ),
            "defect_radius": float(observed_config.defect_radius),
            "contrast_factor": float(
                observed_config.contrast_factor
            ),
            "domain_length_x": float(
                observed_config.domain_length_x
            ),
            "domain_length_y": float(
                observed_config.domain_length_y
            ),
            "dt": float(observed_config.dt),
            "num_steps": int(observed_config.num_steps),
            "final_time": float(observed_config.final_time),
            "every_k": int(resolved_every_k),
            "time_convention": "start_time_plus_snapshot_index_times_dt",
            "exported_first_time": float(
                observed_export["exported_times"][0]
            ),
            "exported_last_time": float(
                observed_export["exported_times"][-1]
            ),
            "num_snapshots_available": int(
                observed_export["num_snapshots_available"]
            ),
            "num_snapshots_exported": int(
                observed_export["num_snapshots_exported"]
            ),
            "exported_indices": list(
                observed_export["exported_indices"]
            ),
            "exported_times": list(
                observed_export["exported_times"]
            ),
            "receiver_box": template_payload.get("receiver_box"),
            "checks": checks,
            "mesh": {
                **dict(spatial_context.mesh_metadata),
                "fingerprint": mesh_fingerprint,
            },
            "configs": {
                "simulated": simulated_config.as_dict(),
                "observed": observed_config.as_dict(),
            },
            "direct_results": {
                "simulated": simulated_summary,
                "observed": observed_summary,
            },
            "exports": {
                "direct_simulated": simulated_export,
                "direct_observed": observed_export,
            },
            "paraview_entrypoints": [
                simulated_export["paraview_entrypoint"],
                observed_export["paraview_entrypoint"],
            ],
            "timings": timings,
            "elapsed_seconds": float(
                time.perf_counter() - started
            ),
        }

        manifest_name = (
            "common_mesh_direct_pair_manifest_"
            f"rank0000of{comm.size:04d}.json"
        )
        manifest_path = output_dir / "manifests" / manifest_name

        if comm.rank == 0:
            write_manifest(manifest_path, manifest)

        comm.Barrier()

        manifest["manifest"] = str(manifest_path)
        return manifest

    finally:
        if simulated_result is not None:
            simulated_result.destroy()

        if observed_result is not None:
            observed_result.destroy()
