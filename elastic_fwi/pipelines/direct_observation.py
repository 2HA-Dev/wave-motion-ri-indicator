from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from elastic_fwi.data.synthetic_observations import SyntheticBoundaryObservationData
from elastic_fwi.data.synthetic_observations import synthetic_data_from_boundary_records
from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    rectangular_pitch_catch_boundary_specs,
    rectangular_pitch_catch_boundary_specs,
    unit_square_pitch_catch_boundary_specs,
)
from elastic_fwi.meshing import (
    plan_wavelength_mesh_sizing,
    RefinementDisk,
    create_gmsh_refined_rectangle_mesh,
)
from elastic_fwi.materials.defect_json import load_defect_scenario_json
from elastic_fwi.materials.dg0_fields import (
    defective_dg0_material_functions,
    homogeneous_dg0_material_functions,
)
from elastic_fwi.materials.localized import localized_disk_dg0_material_functions
from elastic_fwi.materials.orthotropic import (
    OrthotropicPlaneStrainConstants,
    LEGACY_HEALTHY_MATERIAL,
    legacy_defect_material,
)
from elastic_fwi.observations.boundary_observation import (
    gamma1_reflection_observation,
    gamma2_transmission_observation,
)
from elastic_fwi.observations.time_series import evaluate_boundary_observation_time_series
from elastic_fwi.observations.internal_gaussian_array import (
    evaluate_internal_gaussian_array_observation_time_series,
    internal_gaussian_array_observation,
)
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    create_vector_lagrange_space,
)
from elastic_fwi.wave_physics import (
    absorbing_layer_summary_from_function,
    create_absorbing_layer_dg0,
    plan_absorbing_layer,
)

from elastic_fwi.solvers.newmark import LinearSolveConfig
from elastic_fwi.solvers.time_loop import (
    NewmarkTimeLoopConfig,
    run_newmark_time_loop,
)
from elastic_fwi.sources.boundary_arrays import (
    build_boundary_entity,
    default_element_length_for_side,
    lateral_boundary_specs,
)
from elastic_fwi.sources.internal import (
    internal_gaussian_array_hann_source,
    internal_disk_hann_source,
    internal_gaussian_hann_source,
)
from elastic_fwi.sources.traction import gamma1_vertical_traction_source


_ALLOWED_MATERIAL_KINDS = {
    "healthy",
    "homogeneous_contrast",
    "localized_disk_defect",
    "multi_disk_defects",
    "custom_homogeneous",
}


@dataclass(frozen=True)
class DirectObservationPipelineConfig:
    """
    Configuration of the reusable direct observation pipeline.

    material_kind:
        healthy
        homogeneous_contrast
        localized_disk_defect
    """

    nx: int = 2
    ny: int = 2
    degree: int = 1
    mesh_kind: str = "structured"
    mesh_far_h: float | None = None
    mesh_near_h: float | None = None
    mesh_refine_source: bool = False
    mesh_refine_defect: bool = False
    mesh_refine_from_defects_json: bool = False
    mesh_refine_source_radius: float | None = None
    mesh_refine_defect_radius: float | None = None
    mesh_auto_from_wavelength: bool = False
    mesh_target_ppw: float = 10.0
    mesh_near_target_ppw: float = 20.0
    mesh_near_stiffness_factor: float = 1.0
    absorbing_layer: bool = False
    absorbing_layer_width: float | None = None
    absorbing_layer_width_wavelengths: float = 1.5
    absorbing_layer_min_cells: int = 6
    absorbing_layer_strength: float = 3.0
    absorbing_layer_polynomial_order: float = 2.0
    absorbing_layer_max_fraction: float = 0.45

    domain_length_x: float = 1.0
    domain_length_y: float = 1.0
    source_segment_xmin: float = 0.0
    source_segment_xmax: float = 1.0
    receiver_segment_xmin: float = 0.0
    receiver_segment_xmax: float = 1.0


    theta: float = 0.25
    dt: float = 0.1
    num_steps: int = 5
    start_time: float = 0.0
    store_displacement_snapshots: bool = False
    reuse_effective_solver: bool = True

    material_kind: str = "healthy"
    contrast_factor: float = 0.8

    custom_rho: float = LEGACY_HEALTHY_MATERIAL.rho
    custom_C11: float = LEGACY_HEALTHY_MATERIAL.C11
    custom_C22: float = LEGACY_HEALTHY_MATERIAL.C22
    custom_C12: float = LEGACY_HEALTHY_MATERIAL.C12
    custom_C66: float = LEGACY_HEALTHY_MATERIAL.C66

    defect_center_x: float = 0.5
    defect_center_y: float = 0.5
    defect_radius: float = 0.35
    defects_json: str | None = None
    defect_background_factor: float = 1.0
    defect_conflict_policy: str = "last"

    source_kind: str = "boundary_segment"
    source_side: str = "bottom"
    source_center: float | None = None
    source_length: float | None = None
    source_array_start: float | None = None
    source_array_end: float | None = None
    source_array_num_elements: int = 1
    source_array_element_length: float | None = None
    source_array_start_x: float | None = None
    source_array_start_y: float | None = None
    source_array_end_x: float | None = None
    source_array_end_y: float | None = None

    receiver_kind: str = "boundary_segment"
    receiver_side: str = "top"
    receiver_center: float | None = None
    receiver_length: float | None = None
    receiver_array_start: float | None = None
    receiver_array_end: float | None = None
    receiver_array_num_elements: int = 1
    receiver_array_element_length: float | None = None
    receiver_array_start_x: float | None = None
    receiver_array_start_y: float | None = None
    receiver_array_end_x: float | None = None
    receiver_array_end_y: float | None = None
    auto_place_probes_inside_absorbing_layer: bool = False

    source_center_x: float = 0.5
    source_center_y: float = 0.5
    source_radius: float = 0.05
    source_sigma: float = 0.05
    source_direction_x: float = 0.0
    source_direction_y: float = 1.0

    source_amplitude: float = 1.0
    source_frequency: float = 1.0
    source_num_periods: float = 1.0

    ksp_type: str = "cg"
    pc_type: str = "jacobi"
    ksp_rtol: float = 1.0e-10
    ksp_atol: float = 1.0e-12
    ksp_max_it: int = 1000

    def __post_init__(self) -> None:
        nx = int(self.nx)
        ny = int(self.ny)
        degree = int(self.degree)
        mesh_kind = str(self.mesh_kind)
        mesh_far_h = None if self.mesh_far_h is None else float(self.mesh_far_h)
        mesh_near_h = None if self.mesh_near_h is None else float(self.mesh_near_h)
        mesh_refine_source = bool(self.mesh_refine_source)
        mesh_refine_defect = bool(self.mesh_refine_defect)
        mesh_refine_from_defects_json = bool(self.mesh_refine_from_defects_json)
        mesh_refine_source_radius = None if self.mesh_refine_source_radius is None else float(self.mesh_refine_source_radius)
        mesh_refine_defect_radius = None if self.mesh_refine_defect_radius is None else float(self.mesh_refine_defect_radius)
        mesh_auto_from_wavelength = bool(self.mesh_auto_from_wavelength)
        mesh_target_ppw = float(self.mesh_target_ppw)
        mesh_near_target_ppw = float(self.mesh_near_target_ppw)
        mesh_near_stiffness_factor = float(self.mesh_near_stiffness_factor)

        if not isfinite(mesh_near_stiffness_factor) or mesh_near_stiffness_factor <= 0.0:
            raise ValueError("mesh_near_stiffness_factor must be strictly positive and finite.")

        absorbing_layer = bool(self.absorbing_layer)
        absorbing_layer_width = (
            None if self.absorbing_layer_width is None
            else float(self.absorbing_layer_width)
        )
        absorbing_layer_width_wavelengths = float(self.absorbing_layer_width_wavelengths)
        absorbing_layer_min_cells = int(self.absorbing_layer_min_cells)
        absorbing_layer_strength = float(self.absorbing_layer_strength)
        absorbing_layer_polynomial_order = float(self.absorbing_layer_polynomial_order)
        absorbing_layer_max_fraction = float(self.absorbing_layer_max_fraction)

        num_steps = int(self.num_steps)
        ksp_max_it = int(self.ksp_max_it)

        domain_length_x = float(self.domain_length_x)
        domain_length_y = float(self.domain_length_y)
        source_segment_xmin = float(self.source_segment_xmin)
        source_segment_xmax = float(self.source_segment_xmax)
        receiver_segment_xmin = float(self.receiver_segment_xmin)
        receiver_segment_xmax = float(self.receiver_segment_xmax)
        domain_length_x = float(self.domain_length_x)
        domain_length_y = float(self.domain_length_y)
        source_segment_xmin = float(self.source_segment_xmin)
        source_segment_xmax = float(self.source_segment_xmax)
        receiver_segment_xmin = float(self.receiver_segment_xmin)
        receiver_segment_xmax = float(self.receiver_segment_xmax)
        theta = float(self.theta)
        dt = float(self.dt)
        start_time = float(self.start_time)
        store_displacement_snapshots = bool(self.store_displacement_snapshots)
        reuse_effective_solver = bool(self.reuse_effective_solver)
        material_kind = str(self.material_kind)
        contrast_factor = float(self.contrast_factor)
        custom_rho = float(self.custom_rho)
        custom_C11 = float(self.custom_C11)
        custom_C22 = float(self.custom_C22)
        custom_C12 = float(self.custom_C12)
        custom_C66 = float(self.custom_C66)
        defect_center_x = float(self.defect_center_x)
        defect_center_y = float(self.defect_center_y)
        defect_radius = float(self.defect_radius)
        defects_json = self.defects_json
        defect_background_factor = float(self.defect_background_factor)
        defect_conflict_policy = str(self.defect_conflict_policy)
        source_kind = str(self.source_kind)
        source_side = str(self.source_side)
        source_center = None if self.source_center is None else float(self.source_center)
        source_length = None if self.source_length is None else float(self.source_length)
        source_array_start = None if self.source_array_start is None else float(self.source_array_start)
        source_array_end = None if self.source_array_end is None else float(self.source_array_end)
        source_array_num_elements = int(self.source_array_num_elements)
        source_array_element_length = None if self.source_array_element_length is None else float(self.source_array_element_length)
        source_array_start_x = None if self.source_array_start_x is None else float(self.source_array_start_x)
        source_array_start_y = None if self.source_array_start_y is None else float(self.source_array_start_y)
        source_array_end_x = None if self.source_array_end_x is None else float(self.source_array_end_x)
        source_array_end_y = None if self.source_array_end_y is None else float(self.source_array_end_y)

        receiver_kind = str(self.receiver_kind)
        receiver_side = str(self.receiver_side)
        receiver_center = None if self.receiver_center is None else float(self.receiver_center)
        receiver_length = None if self.receiver_length is None else float(self.receiver_length)
        receiver_array_start = None if self.receiver_array_start is None else float(self.receiver_array_start)
        receiver_array_end = None if self.receiver_array_end is None else float(self.receiver_array_end)
        receiver_array_num_elements = int(self.receiver_array_num_elements)
        receiver_array_element_length = None if self.receiver_array_element_length is None else float(self.receiver_array_element_length)
        receiver_array_start_x = None if self.receiver_array_start_x is None else float(self.receiver_array_start_x)
        receiver_array_start_y = None if self.receiver_array_start_y is None else float(self.receiver_array_start_y)
        receiver_array_end_x = None if self.receiver_array_end_x is None else float(self.receiver_array_end_x)
        receiver_array_end_y = None if self.receiver_array_end_y is None else float(self.receiver_array_end_y)
        auto_place_probes_inside_absorbing_layer = bool(self.auto_place_probes_inside_absorbing_layer)

        if auto_place_probes_inside_absorbing_layer:
            if not absorbing_layer:
                raise ValueError(
                    "auto_place_probes_inside_absorbing_layer requires absorbing_layer=True."
                )

            if absorbing_layer_width is None:
                raise ValueError(
                    "auto_place_probes_inside_absorbing_layer currently requires "
                    "an explicit absorbing_layer_width."
                )

            inner_bottom_y = float(absorbing_layer_width)
            inner_top_y = float(domain_length_y - absorbing_layer_width)

            if source_kind == "internal_gaussian_array":
                if source_array_start_x is None or source_array_end_x is None:
                    raise ValueError(
                        "auto-place internal source requires source_array_start_x "
                        "and source_array_end_x."
                    )
                source_array_start_y = inner_bottom_y
                source_array_end_y = inner_bottom_y

            if receiver_kind == "internal_gaussian_array":
                if receiver_array_start_x is None or receiver_array_end_x is None:
                    raise ValueError(
                        "auto-place internal receiver requires receiver_array_start_x "
                        "and receiver_array_end_x."
                    )
                receiver_array_start_y = inner_top_y
                receiver_array_end_y = inner_top_y

        source_center_x = float(self.source_center_x)
        source_center_y = float(self.source_center_y)
        source_radius = float(self.source_radius)
        source_sigma = float(self.source_sigma)
        source_direction_x = float(self.source_direction_x)
        source_direction_y = float(self.source_direction_y)
        source_amplitude = float(self.source_amplitude)
        source_frequency = float(self.source_frequency)
        source_num_periods = float(self.source_num_periods)

        wavelength_mesh_sizing = None
        wavelength_mesh_sizing_metadata = None
        if mesh_auto_from_wavelength:
            if material_kind == "custom_homogeneous":
                sizing_rho = custom_rho
                sizing_C11 = custom_C11
                sizing_C22 = custom_C22
                sizing_C12 = custom_C12
                sizing_C66 = custom_C66
            else:
                sizing_rho = LEGACY_HEALTHY_MATERIAL.rho
                sizing_C11 = LEGACY_HEALTHY_MATERIAL.C11
                sizing_C22 = LEGACY_HEALTHY_MATERIAL.C22
                sizing_C12 = LEGACY_HEALTHY_MATERIAL.C12
                sizing_C66 = LEGACY_HEALTHY_MATERIAL.C66

            wavelength_mesh_sizing = plan_wavelength_mesh_sizing(
                domain_length_x=domain_length_x,
                domain_length_y=domain_length_y,
                source_frequency=source_frequency,
                rho=sizing_rho,
                C11=sizing_C11,
                C22=sizing_C22,
                C12=sizing_C12,
                C66=sizing_C66,
                target_ppw=mesh_target_ppw,
                near_target_ppw=mesh_near_target_ppw,
            )

            wavelength_mesh_sizing_metadata = wavelength_mesh_sizing.to_dict()
            base_mesh_near_h = float(wavelength_mesh_sizing.mesh_near_h)
            adjusted_mesh_near_h = (
                base_mesh_near_h * sqrt(mesh_near_stiffness_factor)
            )

            if mesh_kind == "structured":
                nx = wavelength_mesh_sizing.nx
                ny = wavelength_mesh_sizing.ny
            elif mesh_kind == "gmsh_refined":
                if mesh_far_h is None:
                    mesh_far_h = wavelength_mesh_sizing.mesh_far_h
                if mesh_near_h is None:
                    mesh_near_h = adjusted_mesh_near_h

            effective_mesh_near_h = (
                adjusted_mesh_near_h
                if mesh_near_h is None
                else float(mesh_near_h)
            )
            wavelength_mesh_sizing_metadata.update(
                {
                    "mesh_near_stiffness_factor": mesh_near_stiffness_factor,
                    "base_near_h": float(wavelength_mesh_sizing.near_h),
                    "base_mesh_near_h": base_mesh_near_h,
                    "adjusted_near_h": adjusted_mesh_near_h,
                    "adjusted_mesh_near_h": adjusted_mesh_near_h,
                    "adjusted_near_lambda_min": (
                        float(wavelength_mesh_sizing.lambda_min)
                        * sqrt(mesh_near_stiffness_factor)
                    ),
                    "near_h": effective_mesh_near_h,
                    "mesh_near_h": effective_mesh_near_h,
                }
            )

        ksp_rtol = float(self.ksp_rtol)
        ksp_atol = float(self.ksp_atol)

        if nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be strictly positive.")

        if degree <= 0:
            raise ValueError("degree must be strictly positive.")

        if mesh_kind not in {"structured", "gmsh_refined"}:
            raise ValueError("mesh_kind must be one of: structured, gmsh_refined.")

        if mesh_far_h is not None and (not isfinite(mesh_far_h) or mesh_far_h <= 0.0):
            raise ValueError("mesh_far_h must be strictly positive and finite when provided.")

        if mesh_near_h is not None and (not isfinite(mesh_near_h) or mesh_near_h <= 0.0):
            raise ValueError("mesh_near_h must be strictly positive and finite when provided.")

        if mesh_refine_source_radius is not None and (not isfinite(mesh_refine_source_radius) or mesh_refine_source_radius <= 0.0):
            raise ValueError("mesh_refine_source_radius must be strictly positive and finite when provided.")

        if mesh_refine_defect_radius is not None and (not isfinite(mesh_refine_defect_radius) or mesh_refine_defect_radius <= 0.0):
            raise ValueError("mesh_refine_defect_radius must be strictly positive and finite when provided.")

        if not isfinite(mesh_target_ppw) or mesh_target_ppw <= 0.0:
            raise ValueError("mesh_target_ppw must be strictly positive and finite.")

        if not isfinite(mesh_near_target_ppw) or mesh_near_target_ppw <= 0.0:
            raise ValueError("mesh_near_target_ppw must be strictly positive and finite.")

        if mesh_near_target_ppw < mesh_target_ppw:
            raise ValueError("mesh_near_target_ppw must be greater than or equal to mesh_target_ppw.")

        if mesh_refine_from_defects_json and not mesh_refine_defect:
            raise ValueError(
                "mesh_refine_from_defects_json=True requires mesh_refine_defect=True."
            )

        if mesh_refine_from_defects_json and defects_json is None:
            raise ValueError(
                "mesh_refine_from_defects_json=True requires defects_json."
            )

        if not isfinite(domain_length_x) or domain_length_x <= 0.0:
            raise ValueError("domain_length_x must be finite and strictly positive.")

        if not isfinite(domain_length_y) or domain_length_y <= 0.0:
            raise ValueError("domain_length_y must be finite and strictly positive.")

        if source_segment_xmin < 0.0 or source_segment_xmax > domain_length_x or source_segment_xmin > source_segment_xmax:
            raise ValueError("source segment must satisfy 0 <= xmin <= xmax <= domain_length_x.")

        if receiver_segment_xmin < 0.0 or receiver_segment_xmax > domain_length_x or receiver_segment_xmin > receiver_segment_xmax:
            raise ValueError("receiver segment must satisfy 0 <= xmin <= xmax <= domain_length_x.")

        if not isfinite(domain_length_x) or domain_length_x <= 0.0:
            raise ValueError("domain_length_x must be finite and strictly positive.")

        if not isfinite(domain_length_y) or domain_length_y <= 0.0:
            raise ValueError("domain_length_y must be finite and strictly positive.")

        if source_segment_xmin < 0.0 or source_segment_xmax > domain_length_x or source_segment_xmin > source_segment_xmax:
            raise ValueError("source segment must satisfy 0 <= xmin <= xmax <= domain_length_x.")

        if receiver_segment_xmin < 0.0 or receiver_segment_xmax > domain_length_x or receiver_segment_xmin > receiver_segment_xmax:
            raise ValueError("receiver segment must satisfy 0 <= xmin <= xmax <= domain_length_x.")

        if not isfinite(theta) or theta < 0.0:
            raise ValueError("theta must be finite and non-negative.")

        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive.")

        if num_steps <= 0:
            raise ValueError("num_steps must be strictly positive.")

        if not isfinite(start_time):
            raise ValueError("start_time must be finite.")

        if self.material_kind not in _ALLOWED_MATERIAL_KINDS:
            raise ValueError(
                f"material_kind must be one of {sorted(_ALLOWED_MATERIAL_KINDS)}."
            )

        if not isfinite(contrast_factor) or contrast_factor <= 0.0:
            raise ValueError("contrast_factor must be finite and strictly positive.")

        if not isfinite(custom_rho) or custom_rho <= 0.0:
            raise ValueError("custom_rho must be finite and strictly positive.")

        if not isfinite(custom_C11) or custom_C11 <= 0.0:
            raise ValueError("custom_C11 must be finite and strictly positive.")

        if not isfinite(custom_C22) or custom_C22 <= 0.0:
            raise ValueError("custom_C22 must be finite and strictly positive.")

        if not isfinite(custom_C12):
            raise ValueError("custom_C12 must be finite.")

        if not isfinite(custom_C66) or custom_C66 <= 0.0:
            raise ValueError("custom_C66 must be finite and strictly positive.")

        if not isfinite(defect_center_x) or not isfinite(defect_center_y):
            raise ValueError("defect center coordinates must be finite.")

        if not isfinite(defect_radius) or defect_radius <= 0.0:
            raise ValueError("defect_radius must be finite and strictly positive.")

        if source_kind not in {
            "boundary_segment",
            "boundary_point_array",
            "internal_disk",
            "internal_gaussian",
            "internal_gaussian_array",
        }:
            raise ValueError(
                "source_kind must be one of: boundary_segment, boundary_point_array, "
                "internal_disk, internal_gaussian, internal_gaussian_array."
            )

        if receiver_kind not in {
            "boundary_segment",
            "boundary_point_array",
            "internal_disk",
            "internal_gaussian",
            "internal_gaussian_array",
        }:
            raise ValueError(
                "receiver_kind must be one of: boundary_segment, boundary_point_array, "
                "internal_disk, internal_gaussian, internal_gaussian_array."
            )

        if source_array_num_elements <= 0:
            raise ValueError("source_array_num_elements must be strictly positive.")

        if receiver_array_num_elements <= 0:
            raise ValueError("receiver_array_num_elements must be strictly positive.")

        if not isfinite(source_center_x) or not isfinite(source_center_y):
            raise ValueError("source internal center coordinates must be finite.")

        if not isfinite(source_radius) or source_radius <= 0.0:
            raise ValueError("source_radius must be strictly positive and finite.")

        if not isfinite(source_sigma) or source_sigma <= 0.0:
            raise ValueError("source_sigma must be strictly positive and finite.")


        if receiver_kind == "internal_gaussian_array":
            if auto_place_probes_inside_absorbing_layer:
                if not absorbing_layer:
                    raise ValueError(
                        "auto_place_probes_inside_absorbing_layer requires absorbing_layer=True."
                    )
            else:
                required_receiver_coordinates = {
                    "receiver_array_start_x": receiver_array_start_x,
                    "receiver_array_start_y": receiver_array_start_y,
                    "receiver_array_end_x": receiver_array_end_x,
                    "receiver_array_end_y": receiver_array_end_y,
                }

                missing = [
                    name for name, value in required_receiver_coordinates.items()
                    if value is None
                ]

                if missing:
                    raise ValueError(
                        "internal_gaussian_array receiver requires "
                        + ", ".join(missing)
                        + "."
                    )

                for name, value in required_receiver_coordinates.items():
                    if not isfinite(value):
                        raise ValueError(f"{name} must be finite.")

        if source_kind == "internal_gaussian_array":
            if auto_place_probes_inside_absorbing_layer:
                if not absorbing_layer:
                    raise ValueError(
                        "auto_place_probes_inside_absorbing_layer requires absorbing_layer=True."
                    )
            else:
                required_array_coordinates = {
                    "source_array_start_x": source_array_start_x,
                    "source_array_start_y": source_array_start_y,
                    "source_array_end_x": source_array_end_x,
                    "source_array_end_y": source_array_end_y,
                }

                missing = [
                    name for name, value in required_array_coordinates.items()
                    if value is None
                ]

                if missing:
                    raise ValueError(
                        "internal_gaussian_array requires "
                        + ", ".join(missing)
                        + "."
                    )

                for name, value in required_array_coordinates.items():
                    if not isfinite(value):
                        raise ValueError(f"{name} must be finite.")

            if source_array_num_elements <= 0:
                raise ValueError("source_array_num_elements must be strictly positive.")

            if source_sigma <= 0.0:
                raise ValueError("source_sigma must be strictly positive for internal_gaussian_array.")

        if not isfinite(source_direction_x) or not isfinite(source_direction_y):
            raise ValueError("source direction components must be finite.")

        if source_direction_x == 0.0 and source_direction_y == 0.0:
            raise ValueError("source direction must not be the zero vector.")

        if not isfinite(source_amplitude):
            raise ValueError("source_amplitude must be finite.")

        if not isfinite(source_frequency) or source_frequency <= 0.0:
            raise ValueError("source_frequency must be finite and strictly positive.")

        if not isfinite(source_num_periods) or source_num_periods <= 0.0:
            raise ValueError("source_num_periods must be finite and strictly positive.")

        if not isfinite(ksp_rtol) or ksp_rtol < 0.0:
            raise ValueError("ksp_rtol must be finite and non-negative.")

        if not isfinite(ksp_atol) or ksp_atol < 0.0:
            raise ValueError("ksp_atol must be finite and non-negative.")

        if ksp_max_it <= 0:
            raise ValueError("ksp_max_it must be strictly positive.")

        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "mesh_kind", mesh_kind)
        object.__setattr__(self, "mesh_far_h", mesh_far_h)
        object.__setattr__(self, "mesh_near_h", mesh_near_h)
        object.__setattr__(self, "mesh_refine_source", mesh_refine_source)
        object.__setattr__(self, "mesh_refine_defect", mesh_refine_defect)
        object.__setattr__(self, "mesh_refine_from_defects_json", mesh_refine_from_defects_json)
        object.__setattr__(self, "mesh_refine_source_radius", mesh_refine_source_radius)
        object.__setattr__(self, "mesh_refine_defect_radius", mesh_refine_defect_radius)
        object.__setattr__(self, "mesh_auto_from_wavelength", mesh_auto_from_wavelength)
        object.__setattr__(self, "mesh_target_ppw", mesh_target_ppw)
        object.__setattr__(self, "mesh_near_target_ppw", mesh_near_target_ppw)
        object.__setattr__(self, "mesh_near_stiffness_factor", mesh_near_stiffness_factor)
        object.__setattr__(self, "wavelength_mesh_sizing", wavelength_mesh_sizing_metadata)
        object.__setattr__(self, "domain_length_x", domain_length_x)
        object.__setattr__(self, "domain_length_y", domain_length_y)
        object.__setattr__(self, "source_segment_xmin", source_segment_xmin)
        object.__setattr__(self, "source_segment_xmax", source_segment_xmax)
        object.__setattr__(self, "receiver_segment_xmin", receiver_segment_xmin)
        object.__setattr__(self, "receiver_segment_xmax", receiver_segment_xmax)
        object.__setattr__(self, "domain_length_x", domain_length_x)
        object.__setattr__(self, "domain_length_y", domain_length_y)
        object.__setattr__(self, "source_segment_xmin", source_segment_xmin)
        object.__setattr__(self, "source_segment_xmax", source_segment_xmax)
        object.__setattr__(self, "receiver_segment_xmin", receiver_segment_xmin)
        object.__setattr__(self, "receiver_segment_xmax", receiver_segment_xmax)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "num_steps", num_steps)
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(
            self,
            "store_displacement_snapshots",
            store_displacement_snapshots,
        )
        object.__setattr__(self, "reuse_effective_solver", reuse_effective_solver)
        object.__setattr__(self, "contrast_factor", contrast_factor)
        object.__setattr__(self, "custom_rho", custom_rho)
        object.__setattr__(self, "custom_C11", custom_C11)
        object.__setattr__(self, "custom_C22", custom_C22)
        object.__setattr__(self, "custom_C12", custom_C12)
        object.__setattr__(self, "custom_C66", custom_C66)
        object.__setattr__(self, "defect_center_x", defect_center_x)
        object.__setattr__(self, "defect_center_y", defect_center_y)
        object.__setattr__(self, "defect_radius", defect_radius)
        object.__setattr__(self, "defects_json", defects_json)
        object.__setattr__(self, "defect_background_factor", defect_background_factor)
        object.__setattr__(self, "defect_conflict_policy", defect_conflict_policy)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_side", source_side)
        object.__setattr__(self, "source_center", source_center)
        object.__setattr__(self, "source_length", source_length)
        object.__setattr__(self, "source_array_start", source_array_start)
        object.__setattr__(self, "source_array_end", source_array_end)
        object.__setattr__(self, "source_array_num_elements", source_array_num_elements)
        object.__setattr__(self, "source_array_element_length", source_array_element_length)
        object.__setattr__(self, "source_array_start_x", source_array_start_x)
        object.__setattr__(self, "source_array_start_y", source_array_start_y)
        object.__setattr__(self, "source_array_end_x", source_array_end_x)
        object.__setattr__(self, "source_array_end_y", source_array_end_y)
        object.__setattr__(self, "receiver_kind", receiver_kind)
        object.__setattr__(self, "receiver_side", receiver_side)
        object.__setattr__(self, "receiver_center", receiver_center)
        object.__setattr__(self, "receiver_length", receiver_length)
        object.__setattr__(self, "receiver_array_start", receiver_array_start)
        object.__setattr__(self, "receiver_array_end", receiver_array_end)
        object.__setattr__(self, "receiver_array_num_elements", receiver_array_num_elements)
        object.__setattr__(self, "receiver_array_element_length", receiver_array_element_length)
        object.__setattr__(self, "receiver_array_start_x", receiver_array_start_x)
        object.__setattr__(self, "receiver_array_start_y", receiver_array_start_y)
        object.__setattr__(self, "receiver_array_end_x", receiver_array_end_x)
        object.__setattr__(self, "receiver_array_end_y", receiver_array_end_y)
        object.__setattr__(self, "auto_place_probes_inside_absorbing_layer", auto_place_probes_inside_absorbing_layer)
        object.__setattr__(self, "source_center_x", source_center_x)
        object.__setattr__(self, "source_center_y", source_center_y)
        object.__setattr__(self, "source_radius", source_radius)
        object.__setattr__(self, "source_sigma", source_sigma)
        object.__setattr__(self, "source_direction_x", source_direction_x)
        object.__setattr__(self, "source_direction_y", source_direction_y)
        object.__setattr__(self, "source_amplitude", source_amplitude)
        object.__setattr__(self, "source_frequency", source_frequency)
        object.__setattr__(self, "source_num_periods", source_num_periods)
        object.__setattr__(self, "ksp_rtol", ksp_rtol)
        object.__setattr__(self, "ksp_atol", ksp_atol)
        object.__setattr__(self, "ksp_max_it", ksp_max_it)

    @property
    def final_time(self) -> float:
        return float(self.start_time + self.num_steps * self.dt)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "degree": self.degree,
            "mesh_kind": self.mesh_kind,
            "mesh_far_h": self.mesh_far_h,
            "mesh_near_h": self.mesh_near_h,
            "mesh_refine_source": self.mesh_refine_source,
            "mesh_refine_defect": self.mesh_refine_defect,
            "mesh_refine_from_defects_json": self.mesh_refine_from_defects_json,
            "mesh_refine_source_radius": self.mesh_refine_source_radius,
            "mesh_refine_defect_radius": self.mesh_refine_defect_radius,
            "mesh_auto_from_wavelength": self.mesh_auto_from_wavelength,
            "mesh_target_ppw": self.mesh_target_ppw,
            "mesh_near_target_ppw": self.mesh_near_target_ppw,
            "mesh_near_stiffness_factor": self.mesh_near_stiffness_factor,
            "wavelength_mesh_sizing": self.wavelength_mesh_sizing,
            "domain_length_x": self.domain_length_x,
            "domain_length_y": self.domain_length_y,
            "source_segment_xmin": self.source_segment_xmin,
            "source_segment_xmax": self.source_segment_xmax,
            "receiver_segment_xmin": self.receiver_segment_xmin,
            "receiver_segment_xmax": self.receiver_segment_xmax,
            "domain_length_x": self.domain_length_x,
            "domain_length_y": self.domain_length_y,
            "source_segment_xmin": self.source_segment_xmin,
            "source_segment_xmax": self.source_segment_xmax,
            "receiver_segment_xmin": self.receiver_segment_xmin,
            "receiver_segment_xmax": self.receiver_segment_xmax,
            "theta": self.theta,
            "dt": self.dt,
            "num_steps": self.num_steps,
            "start_time": self.start_time,
            "final_time": self.final_time,
            "store_displacement_snapshots": self.store_displacement_snapshots,
            "reuse_effective_solver": self.reuse_effective_solver,
            "material_kind": self.material_kind,
            "contrast_factor": self.contrast_factor,
            "custom_rho": self.custom_rho,
            "custom_C11": self.custom_C11,
            "custom_C22": self.custom_C22,
            "custom_C12": self.custom_C12,
            "custom_C66": self.custom_C66,
            "defect_center_x": self.defect_center_x,
            "defect_center_y": self.defect_center_y,
            "defect_radius": self.defect_radius,
            "defects_json": self.defects_json,
            "defect_background_factor": self.defect_background_factor,
            "defect_conflict_policy": self.defect_conflict_policy,
            "source_kind": self.source_kind,
            "source_side": self.source_side,
            "source_center": self.source_center,
            "source_length": self.source_length,
            "source_array_start": self.source_array_start,
            "source_array_end": self.source_array_end,
            "source_array_num_elements": self.source_array_num_elements,
            "source_array_element_length": self.source_array_element_length,
            "source_array_start_x": self.source_array_start_x,
            "source_array_start_y": self.source_array_start_y,
            "source_array_end_x": self.source_array_end_x,
            "source_array_end_y": self.source_array_end_y,
            "receiver_kind": self.receiver_kind,
            "receiver_side": self.receiver_side,
            "receiver_center": self.receiver_center,
            "receiver_length": self.receiver_length,
            "receiver_array_start": self.receiver_array_start,
            "receiver_array_end": self.receiver_array_end,
            "receiver_array_num_elements": self.receiver_array_num_elements,
            "receiver_array_element_length": self.receiver_array_element_length,
            "receiver_array_start_x": self.receiver_array_start_x,
            "receiver_array_start_y": self.receiver_array_start_y,
            "receiver_array_end_x": self.receiver_array_end_x,
            "receiver_array_end_y": self.receiver_array_end_y,
            "auto_place_probes_inside_absorbing_layer": self.auto_place_probes_inside_absorbing_layer,
            "source_center_x": self.source_center_x,
            "source_center_y": self.source_center_y,
            "source_radius": self.source_radius,
            "source_sigma": self.source_sigma,
            "source_direction_x": self.source_direction_x,
            "source_direction_y": self.source_direction_y,
            "source_amplitude": self.source_amplitude,
            "source_frequency": self.source_frequency,
            "source_num_periods": self.source_num_periods,
            "ksp_type": self.ksp_type,
            "pc_type": self.pc_type,
            "ksp_rtol": self.ksp_rtol,
            "ksp_atol": self.ksp_atol,
            "ksp_max_it": self.ksp_max_it,
        }


@dataclass(frozen=True)
class DirectObservationPipelineResult:
    config: DirectObservationPipelineConfig
    data: SyntheticBoundaryObservationData
    time_loop_records: list[dict[str, Any]]
    mesh_metadata: dict[str, Any]
    material_metadata: dict[str, Any]
    source_metadata: dict[str, Any]
    observation_metadata: dict[str, Any]
    mpi_size: int
    displacement_snapshots: tuple[Any, ...] = ()
    function_space: Any | None = None
    operator_config: ElasticOperatorConfig | None = None

    def destroy(self) -> None:
        for vector in self.displacement_snapshots:
            vector.destroy()

    @property
    def num_displacement_snapshots(self) -> int:
        return int(len(self.displacement_snapshots))

    @property
    def has_displacement_snapshots(self) -> bool:
        return bool(self.displacement_snapshots)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "direct_observation_pipeline_result",
            "mpi_size": int(self.mpi_size),
            "config": self.config.as_dict(),
            "mesh": dict(self.mesh_metadata),
            "material": dict(self.material_metadata),
            "source": dict(self.source_metadata),
            "observations": dict(self.observation_metadata),
            "state": {
                "displacement_snapshots_stored": self.has_displacement_snapshots,
                "num_displacement_snapshots": self.num_displacement_snapshots,
            },
            "data": self.data.as_dict(),
            "time_loop_records": list(self.time_loop_records),
            "values": {
                "max_reflection": self.data.reflection.max_value,
                "max_transmission": self.data.transmission.max_value,
                "final_reflection": self.data.reflection.final_value,
                "final_transmission": self.data.transmission.final_value,
            },
        }


def _build_source_receiver_boundary_specs(config: DirectObservationPipelineConfig):
    source_specs = ()

    if config.source_kind in {"boundary_segment", "boundary_point_array"}:
        source_default_element_length = default_element_length_for_side(
            side=config.source_side,
            domain_length_x=config.domain_length_x,
            domain_length_y=config.domain_length_y,
            nx=config.nx,
            ny=config.ny,
        )

        source_entity = build_boundary_entity(
            kind=config.source_kind,
            name="gamma1_source",
            side=config.source_side,
            tag=1,
            center=config.source_center,
            length=config.source_length,
            array_start=config.source_array_start,
            array_end=config.source_array_end,
            array_num_elements=config.source_array_num_elements,
            array_element_length=config.source_array_element_length,
            fallback_interval_min=config.source_segment_xmin,
            fallback_interval_max=config.source_segment_xmax,
            default_element_length=source_default_element_length,
        )

        source_specs = source_entity.boundary_specs(
            domain_length_x=config.domain_length_x,
            domain_length_y=config.domain_length_y,
        )
        source_geometry = source_entity.metadata()

    else:
        source_geometry = {
            "kind": config.source_kind,
            "boundary_tag": None,
            "note": "Internal source: no Gamma1 boundary facets are tagged.",
            "array_start_x": config.source_array_start_x,
            "array_start_y": config.source_array_start_y,
            "array_end_x": config.source_array_end_x,
            "array_end_y": config.source_array_end_y,
            "array_num_elements": config.source_array_num_elements,
            "sigma": config.source_sigma,
        }

    receiver_specs = ()

    if config.receiver_kind in {"boundary_segment", "boundary_point_array"}:
        receiver_default_element_length = default_element_length_for_side(
            side=config.receiver_side,
            domain_length_x=config.domain_length_x,
            domain_length_y=config.domain_length_y,
            nx=config.nx,
            ny=config.ny,
        )

        receiver_entity = build_boundary_entity(
            kind=config.receiver_kind,
            name="gamma2_receiver",
            side=config.receiver_side,
            tag=2,
            center=config.receiver_center,
            length=config.receiver_length,
            array_start=config.receiver_array_start,
            array_end=config.receiver_array_end,
            array_num_elements=config.receiver_array_num_elements,
            array_element_length=config.receiver_array_element_length,
            fallback_interval_min=config.receiver_segment_xmin,
            fallback_interval_max=config.receiver_segment_xmax,
            default_element_length=receiver_default_element_length,
        )

        receiver_specs = receiver_entity.boundary_specs(
            domain_length_x=config.domain_length_x,
            domain_length_y=config.domain_length_y,
        )
        receiver_geometry = receiver_entity.metadata()

    else:
        receiver_geometry = {
            "kind": config.receiver_kind,
            "boundary_tag": None,
            "note": "Internal receiver: no Gamma2 boundary facets are tagged.",
            "array_start_x": config.receiver_array_start_x,
            "array_start_y": config.receiver_array_start_y,
            "array_end_x": config.receiver_array_end_x,
            "array_end_y": config.receiver_array_end_y,
            "array_num_elements": config.receiver_array_num_elements,
            "element_length": config.receiver_array_element_length,
        }

    boundary_specs = (
        *source_specs,
        *receiver_specs,
        *lateral_boundary_specs(
            domain_length_x=config.domain_length_x,
            domain_length_y=config.domain_length_y,
        ),
    )

    metadata = {
        "source_geometry": source_geometry,
        "receiver_geometry": receiver_geometry,
    }

    return boundary_specs, metadata


def _build_direct_source(config: DirectObservationPipelineConfig):
    direction = (config.source_direction_x, config.source_direction_y)

    if config.source_kind == "internal_disk":
        source = internal_disk_hann_source(
            center_x=config.source_center_x,
            center_y=config.source_center_y,
            radius=config.source_radius,
            direction=direction,
            amplitude=config.source_amplitude,
            frequency=config.source_frequency,
            t0=config.start_time,
            num_periods=config.source_num_periods,
        )
        return source, source.metadata()

    if config.source_kind == "internal_gaussian":
        source = internal_gaussian_hann_source(
            center_x=config.source_center_x,
            center_y=config.source_center_y,
            sigma=config.source_sigma,
            direction=direction,
            amplitude=config.source_amplitude,
            frequency=config.source_frequency,
            t0=config.start_time,
            num_periods=config.source_num_periods,
        )
        return source, source.metadata()

    if config.source_kind == "internal_gaussian_array":
        source = internal_gaussian_array_hann_source(
            start_x=config.source_array_start_x,
            start_y=config.source_array_start_y,
            end_x=config.source_array_end_x,
            end_y=config.source_array_end_y,
            num_elements=config.source_array_num_elements,
            sigma=config.source_sigma,
            direction=(config.source_direction_x, config.source_direction_y),
            amplitude=config.source_amplitude,
            frequency=config.source_frequency,
            t0=config.start_time,
            num_periods=config.source_num_periods,
        )
        return source, source.metadata()

    source = gamma1_vertical_traction_source(
        tag=1,
        amplitude=config.source_amplitude,
        frequency=config.source_frequency,
        t0=config.start_time,
        num_periods=config.source_num_periods,
    )

    metadata = {
        "kind": "boundary_traction",
        "name": source.name,
        "tag": source.tag,
        "direction": list(source.direction),
    }

    return source, metadata


def _build_materials(msh, config: DirectObservationPipelineConfig):
    if config.material_kind == "healthy":
        materials = homogeneous_dg0_material_functions(
            msh,
            LEGACY_HEALTHY_MATERIAL,
        )
        metadata = {
            "kind": "healthy",
            "constants": {
                "rho": LEGACY_HEALTHY_MATERIAL.rho,
                "C11": LEGACY_HEALTHY_MATERIAL.C11,
                "C22": LEGACY_HEALTHY_MATERIAL.C22,
                "C12": LEGACY_HEALTHY_MATERIAL.C12,
                "C66": LEGACY_HEALTHY_MATERIAL.C66,
            },
        }
        return materials, metadata

    if config.material_kind == "homogeneous_contrast":
        material = legacy_defect_material(factor=config.contrast_factor)
        materials = homogeneous_dg0_material_functions(msh, material)
        metadata = {
            "kind": "homogeneous_contrast",
            "contrast_factor": config.contrast_factor,
            "constants": {
                "rho": material.rho,
                "C11": material.C11,
                "C22": material.C22,
                "C12": material.C12,
                "C66": material.C66,
            },
        }
        return materials, metadata

    if config.material_kind == "custom_homogeneous":
        material = OrthotropicPlaneStrainConstants(
            rho=config.custom_rho,
            C11=config.custom_C11,
            C22=config.custom_C22,
            C12=config.custom_C12,
            C66=config.custom_C66,
        )
        materials = homogeneous_dg0_material_functions(msh, material)
        metadata = {
            "kind": "custom_homogeneous",
            "constants": {
                "rho": material.rho,
                "C11": material.C11,
                "C22": material.C22,
                "C12": material.C12,
                "C66": material.C66,
            },
        }
        return materials, metadata

    if config.material_kind == "localized_disk_defect":
        return localized_disk_dg0_material_functions(
            msh,
            center_x=config.defect_center_x,
            center_y=config.defect_center_y,
            radius=config.defect_radius,
            contrast_factor=config.contrast_factor,
            healthy_material=LEGACY_HEALTHY_MATERIAL,
        )

    if config.material_kind == "multi_disk_defects":
        if config.defects_json is None:
            raise ValueError("material_kind='multi_disk_defects' requires defects_json.")

        scenario, json_metadata = load_defect_scenario_json(config.defects_json)

        materials = defective_dg0_material_functions(
            msh,
            scenario,
            healthy_material=LEGACY_HEALTHY_MATERIAL,
            background_factor=config.defect_background_factor,
            conflict_policy=config.defect_conflict_policy,
        )

        metadata = {
            "kind": "multi_disk_defects",
            "scenario": json_metadata,
            "background_factor": config.defect_background_factor,
            "conflict_policy": config.defect_conflict_policy,
        }

        return materials, metadata

    raise ValueError(f"Unsupported material_kind: {config.material_kind}")


def _effective_mesh_nx_ny(config: DirectObservationPipelineConfig) -> tuple[int, int]:
    return int(config.nx), int(config.ny)


def _default_mesh_far_h(config: DirectObservationPipelineConfig) -> float:
    if config.mesh_auto_from_wavelength and config.wavelength_mesh_sizing is not None:
        return float(config.wavelength_mesh_sizing["mesh_far_h"])

    return max(
        float(config.domain_length_x) / float(config.nx),
        float(config.domain_length_y) / float(config.ny),
    )


def _default_mesh_near_h(config: DirectObservationPipelineConfig) -> float:
    if config.mesh_auto_from_wavelength and config.wavelength_mesh_sizing is not None:
        return float(config.wavelength_mesh_sizing["mesh_near_h"])

    return 0.5 * _default_mesh_far_h(config)


def _build_gmsh_refinement_disks(config: DirectObservationPipelineConfig) -> tuple[RefinementDisk, ...]:
    near_h = config.mesh_near_h
    if near_h is None:
        near_h = _default_mesh_near_h(config)

    disks: list[RefinementDisk] = []

    if config.mesh_refine_source and config.source_kind in {"internal_disk", "internal_gaussian"}:
        if config.mesh_refine_source_radius is not None:
            radius = config.mesh_refine_source_radius
        elif config.source_kind == "internal_disk":
            radius = 3.0 * config.source_radius
        else:
            radius = 3.0 * config.source_sigma

        disks.append(
            RefinementDisk(
                name="source",
                center_x=config.source_center_x,
                center_y=config.source_center_y,
                radius=radius,
                target_h=near_h,
            )
        )

    if config.mesh_refine_defect:
        use_defects_json = (
            config.mesh_refine_from_defects_json
            or config.material_kind == "multi_disk_defects"
        )

        if use_defects_json and config.defects_json is not None:
            _, metadata = load_defect_scenario_json(config.defects_json)
            for index, defect in enumerate(metadata["defects"]):
                defect_radius = float(defect["radius"])
                radius = (
                    config.mesh_refine_defect_radius
                    if config.mesh_refine_defect_radius is not None
                    else 3.0 * defect_radius
                )
                disks.append(
                    RefinementDisk(
                        name=f"defect_{index}",
                        center_x=float(defect["center_x"]),
                        center_y=float(defect["center_y"]),
                        radius=radius,
                        target_h=near_h,
                    )
                )

        elif not use_defects_json:
            # The defect refinement is a property of the numerical experiment
            # geometry, not of the material assigned to the current wave solve.
            # Therefore the healthy/background solve keeps the same refinement
            # disk as the defective solve whenever mesh_refine_defect=True.
            radius = (
                config.mesh_refine_defect_radius
                if config.mesh_refine_defect_radius is not None
                else 3.0 * config.defect_radius
            )
            disks.append(
                RefinementDisk(
                    name="defect",
                    center_x=config.defect_center_x,
                    center_y=config.defect_center_y,
                    radius=radius,
                    target_h=near_h,
                )
            )

    return tuple(disks)


def _create_direct_mesh(config: DirectObservationPipelineConfig, comm: MPI.Comm):


    if config.mesh_kind == "structured":
        return mesh.create_rectangle(
            comm,
            np.array(
                [[0.0, 0.0], [config.domain_length_x, config.domain_length_y]],
                dtype=np.float64,
            ),
            [config.nx, config.ny],
            cell_type=mesh.CellType.triangle,
        )

    if config.mesh_kind == "gmsh_refined":
        far_h = config.mesh_far_h
        if far_h is None:
            far_h = _default_mesh_far_h(config)

        return create_gmsh_refined_rectangle_mesh(
            comm=comm,
            domain_length_x=config.domain_length_x,
            domain_length_y=config.domain_length_y,
            far_h=far_h,
            refinement_disks=_build_gmsh_refinement_disks(config),
        )

    raise ValueError(f"Unsupported mesh_kind: {config.mesh_kind}")



def _absorbing_layer_material_constants(config: DirectObservationPipelineConfig) -> tuple[float, float, float, float]:
    if config.material_kind == "custom_homogeneous":
        return (
            float(config.custom_rho),
            float(config.custom_C11),
            float(config.custom_C22),
            float(config.custom_C66),
        )

    return (
        float(LEGACY_HEALTHY_MATERIAL.rho),
        float(LEGACY_HEALTHY_MATERIAL.C11),
        float(LEGACY_HEALTHY_MATERIAL.C22),
        float(LEGACY_HEALTHY_MATERIAL.C66),
    )


def _absorbing_layer_h_reference(config: DirectObservationPipelineConfig) -> float:
    if config.mesh_kind == "gmsh_refined" and config.mesh_far_h is not None:
        return float(config.mesh_far_h)

    return max(
        float(config.domain_length_x) / float(config.nx),
        float(config.domain_length_y) / float(config.ny),
    )


def _build_absorbing_layer_if_requested(msh, config: DirectObservationPipelineConfig):
    if not config.absorbing_layer:
        return None, {
            "enabled": False,
            "kind": "none",
        }

    rho, C11, C22, C66 = _absorbing_layer_material_constants(config)

    plan = plan_absorbing_layer(
        domain_length_x=config.domain_length_x,
        domain_length_y=config.domain_length_y,
        source_frequency=config.source_frequency,
        rho=rho,
        C11=C11,
        C22=C22,
        C66=C66,
        h_reference=_absorbing_layer_h_reference(config),
        width=config.absorbing_layer_width,
        width_wavelengths=config.absorbing_layer_width_wavelengths,
        min_cells=config.absorbing_layer_min_cells,
        max_fraction=config.absorbing_layer_max_fraction,
        strength=config.absorbing_layer_strength,
        polynomial_order=config.absorbing_layer_polynomial_order,
    )

    sigma = create_absorbing_layer_dg0(
        msh,
        domain_length_x=config.domain_length_x,
        domain_length_y=config.domain_length_y,
        plan=plan,
    )

    return sigma, {
        "enabled": True,
        "kind": "absorbing_sponge_layer",
        "plan": plan.to_dict(),
        "field_summary": absorbing_layer_summary_from_function(sigma),
    }

@dataclass(frozen=True)
class DirectObservationSpatialContext:
    """
    Spatial objects shared by several direct solves.

    The mesh, MPI partition, facet tags, finite-element space and absorbing
    layer are created once. Material fields and operator matrices remain
    specific to each direct solve.
    """

    config: DirectObservationPipelineConfig
    mesh: Any
    facet_tags: Any
    function_space: Any
    absorbing_sigma: Any | None
    absorbing_layer_metadata: dict[str, Any]
    source_receiver_geometry_metadata: dict[str, Any]
    mesh_metadata: dict[str, Any]
    mpi_size: int


_COMMON_MESH_MATERIAL_FIELDS = {
    "material_kind",
    "contrast_factor",
    "custom_rho",
    "custom_C11",
    "custom_C22",
    "custom_C12",
    "custom_C66",
    "defect_background_factor",
    "defect_conflict_policy",
    # Snapshot retention changes memory/output policy only; it must not force
    # creation of a second spatial discretization.
    "store_displacement_snapshots",
}


def direct_observation_common_mesh_mismatches(
    reference: DirectObservationPipelineConfig,
    candidate: DirectObservationPipelineConfig,
) -> dict[str, tuple[Any, Any]]:
    """
    Return all non-material differences between two direct configurations.

    A common-mesh pair must use identical experiment geometry, including the
    declared defect/refinement geometry, discretization, source, receiver,
    time integration and linear-solver parameters. Material coefficients and
    snapshot-retention policy may differ without requiring a second spatial
    discretization.
    """

    reference_values = reference.as_dict()
    candidate_values = candidate.as_dict()

    keys = sorted(
        (set(reference_values) | set(candidate_values))
        - _COMMON_MESH_MATERIAL_FIELDS
    )

    return {
        key: (reference_values.get(key), candidate_values.get(key))
        for key in keys
        if reference_values.get(key) != candidate_values.get(key)
    }


def validate_direct_observation_common_mesh_compatibility(
    reference: DirectObservationPipelineConfig,
    candidate: DirectObservationPipelineConfig,
) -> None:
    mismatches = direct_observation_common_mesh_mismatches(
        reference,
        candidate,
    )

    if mismatches:
        details = ", ".join(
            f"{name}: reference={values[0]!r}, candidate={values[1]!r}"
            for name, values in mismatches.items()
        )
        raise ValueError(
            "Incompatible direct configurations for a common-mesh pair: "
            + details
        )


def build_direct_observation_spatial_context(
    config: DirectObservationPipelineConfig,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> DirectObservationSpatialContext:
    """
    Build the mesh-dependent objects used by one or several direct solves.
    """

    msh = _create_direct_mesh(config, comm)

    absorbing_sigma, absorbing_layer_metadata = (
        _build_absorbing_layer_if_requested(
            msh,
            config,
        )
    )

    boundary_specs, source_receiver_geometry_metadata = (
        _build_source_receiver_boundary_specs(config)
    )

    facet_tags = build_boundary_meshtags(msh, boundary_specs)
    V = create_vector_lagrange_space(msh, degree=config.degree)

    refinement_disks = tuple(_build_gmsh_refinement_disks(config))

    mesh_metadata = {
        "kind": "rectangle",
        "domain_length_x": config.domain_length_x,
        "domain_length_y": config.domain_length_y,
        "nx": config.nx,
        "ny": config.ny,
        "degree": config.degree,
        "mesh_kind": config.mesh_kind,
        "effective_nx": int(config.nx),
        "effective_ny": int(config.ny),
        "mesh_auto_from_wavelength": config.mesh_auto_from_wavelength,
        "mesh_target_ppw": config.mesh_target_ppw,
        "mesh_near_target_ppw": config.mesh_near_target_ppw,
        "mesh_near_stiffness_factor": config.mesh_near_stiffness_factor,
        "wavelength_mesh_sizing": config.wavelength_mesh_sizing,
        "mesh_far_h": config.mesh_far_h,
        "mesh_near_h": config.mesh_near_h,
        "mesh_refine_source": config.mesh_refine_source,
        "mesh_refine_defect": config.mesh_refine_defect,
        "mesh_refine_from_defects_json": (
            config.mesh_refine_from_defects_json
        ),
        "mesh_refine_source_radius": config.mesh_refine_source_radius,
        "mesh_refine_defect_radius": config.mesh_refine_defect_radius,
        "refinement_disks": [
            disk.as_dict() for disk in refinement_disks
        ],
        "topological_dimension": int(msh.topology.dim),
        "geometric_dimension": int(msh.geometry.dim),
        "absorbing_layer": absorbing_layer_metadata,
    }

    return DirectObservationSpatialContext(
        config=config,
        mesh=msh,
        facet_tags=facet_tags,
        function_space=V,
        absorbing_sigma=absorbing_sigma,
        absorbing_layer_metadata=absorbing_layer_metadata,
        source_receiver_geometry_metadata=(
            source_receiver_geometry_metadata
        ),
        mesh_metadata=mesh_metadata,
        mpi_size=int(comm.size),
    )


def run_direct_observation_pipeline(
    config: DirectObservationPipelineConfig | None = None,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
    spatial_context: DirectObservationSpatialContext | None = None,
) -> DirectObservationPipelineResult:
    """
    Run the reusable direct pipeline up to Gamma1/Gamma2 synthetic observations.

    This function does not compute an adjoint or a gradient.
    It is the common forward-observation block used by diagnostics,
    synthetic data generation and inverse routines.
    """

    if config is None:
        config = DirectObservationPipelineConfig()

    if spatial_context is None:
        spatial_context = build_direct_observation_spatial_context(
            config,
            comm=comm,
        )
    else:
        if int(spatial_context.mpi_size) != int(comm.size):
            raise ValueError(
                "The spatial context MPI size does not match the active "
                "communicator."
            )

        validate_direct_observation_common_mesh_compatibility(
            spatial_context.config,
            config,
        )

    msh = spatial_context.mesh
    absorbing_sigma = spatial_context.absorbing_sigma
    absorbing_layer_metadata = (
        spatial_context.absorbing_layer_metadata
    )
    source_receiver_geometry_metadata = (
        spatial_context.source_receiver_geometry_metadata
    )
    facet_tags = spatial_context.facet_tags
    V = spatial_context.function_space

    materials, material_metadata = _build_materials(msh, config)

    matrices = None
    loop_result = None

    try:
        matrices = assemble_elastic_operator_matrices(
            V,
            materials,
            ElasticOperatorConfig(theta=config.theta, dt=config.dt),
            damping_coefficient=absorbing_sigma,
        )


        source, source_geometry_metadata = _build_direct_source(config)

        loop_result = run_newmark_time_loop(
            matrices=matrices,
            V=V,
            facet_tags=facet_tags,
            source=source,
            config=NewmarkTimeLoopConfig(
                num_steps=config.num_steps,
                start_time=config.start_time,
                store_displacements=True,
                reuse_effective_solver=config.reuse_effective_solver,
            ),
            solve_config=LinearSolveConfig(
                ksp_type=config.ksp_type,
                pc_type=config.pc_type,
                rtol=config.ksp_rtol,
                atol=config.ksp_atol,
                max_it=config.ksp_max_it,
            ),
        )

        snapshots = [
            (record.step, record.time, vector)
            for record, vector in zip(
                loop_result.records,
                loop_result.displacement_snapshots,
                strict=True,
            )
        ]

        gamma1 = gamma1_reflection_observation(component="vector")

        gamma1_records = evaluate_boundary_observation_time_series(
            V,
            snapshots,
            facet_tags,
            gamma1,
        )

        if config.receiver_kind == "internal_gaussian_array":
            receiver_sigma = (
                config.receiver_array_element_length
                if config.receiver_array_element_length is not None
                else config.source_sigma
            )

            gamma2_internal = internal_gaussian_array_observation(
                start_x=config.receiver_array_start_x,
                start_y=config.receiver_array_start_y,
                end_x=config.receiver_array_end_x,
                end_y=config.receiver_array_end_y,
                num_elements=config.receiver_array_num_elements,
                sigma=receiver_sigma,
                component="vector",
                tag=2,
            )

            gamma2 = gamma2_internal

            gamma2_records = evaluate_internal_gaussian_array_observation_time_series(
                V,
                snapshots,
                gamma2_internal,
            )
        else:
            gamma2 = gamma2_transmission_observation(component="vector")

            gamma2_records = evaluate_boundary_observation_time_series(
                V,
                snapshots,
                facet_tags,
                gamma2,
            )

        time_loop_records = [record.as_dict() for record in loop_result.records]

        if config.store_displacement_snapshots:
            retained_displacement_snapshots = tuple(
                vector.copy() for vector in loop_result.displacement_snapshots
            )
        else:
            retained_displacement_snapshots = ()

        data = synthetic_data_from_boundary_records(
            reflection_records=gamma1_records,
            transmission_records=gamma2_records,
            metadata={
                "source": "direct_observation_pipeline",
                "material_kind": config.material_kind,
                "theta": config.theta,
                "dt": config.dt,
                "num_steps": config.num_steps,
                "final_time": config.final_time,
                **material_metadata,
            },
        )

        mesh_metadata = dict(spatial_context.mesh_metadata)

        source_metadata = {
            "name": source.name,
            "tag": getattr(source, "tag", None),
            "amplitude": config.source_amplitude,
            "frequency": config.source_frequency,
            "num_periods": config.source_num_periods,
            "geometry": (
                source_geometry_metadata
                if config.source_kind in {"internal_disk", "internal_gaussian"}
                else source_receiver_geometry_metadata["source_geometry"]
            ),
            "segment_xmin": config.source_segment_xmin,
            "segment_xmax": config.source_segment_xmax,
            "segment_y": 0.0,
            "geometry": source_geometry_metadata,
            "segment_xmin": config.source_segment_xmin,
            "segment_xmax": config.source_segment_xmax,
            "segment_y": 0.0,
        }

        if config.source_kind in {"boundary_segment", "boundary_point_array"}:
            source_metadata["geometry"] = source_receiver_geometry_metadata["source_geometry"]
        elif config.source_kind in {"internal_disk", "internal_gaussian"}:
            source_metadata["geometry"] = source_geometry_metadata

        observation_metadata = {
            "reflection": {
                "name": gamma1.name,
                "tag": gamma1.tag,
                "component": gamma1.component,
            },
            "transmission": {
                "name": gamma2.name,
                "tag": gamma2.tag,
                "component": gamma2.component,
                "segment_xmin": config.receiver_segment_xmin,
                "segment_xmax": config.receiver_segment_xmax,
                "segment_y": config.domain_length_y,
                "geometry": source_receiver_geometry_metadata["receiver_geometry"],
            },
        }

        return DirectObservationPipelineResult(
            config=config,
            data=data,
            time_loop_records=time_loop_records,
            mesh_metadata=mesh_metadata,
            material_metadata=material_metadata,
            source_metadata=source_metadata,
            observation_metadata=observation_metadata,
            mpi_size=comm.size,
            displacement_snapshots=retained_displacement_snapshots,
            function_space=V,
            operator_config=ElasticOperatorConfig(theta=config.theta, dt=config.dt),
        )

    finally:
        if loop_result is not None:
            loop_result.destroy()

        if matrices is not None:
            matrices.destroy()
