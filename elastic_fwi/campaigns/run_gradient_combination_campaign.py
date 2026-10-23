from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from mpi4py import MPI

from elastic_fwi.diagnostics.export_inverse_time_series_xdmf import (
    export_inverse_time_series_xdmf,
)


REQUIRED_COMBINATIONS = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
    "residual_interaction",
)

REQUIRED_GRADIENT_FIELDS = (
    "gradient_rho",
    "gradient_C11",
    "gradient_C22",
    "gradient_C12",
    "gradient_C66",
)

REQUIRED_KERNEL_FIELDS = (
    "residual_interaction_rho",
    "residual_interaction_C11",
    "residual_interaction_C22",
    "residual_interaction_C12",
    "residual_interaction_C66",
)


def _required_fields_for_combination(combination: str) -> tuple[str, ...]:
    if str(combination) == "residual_interaction":
        return REQUIRED_KERNEL_FIELDS
    return REQUIRED_GRADIENT_FIELDS


def _article_face_to_face_r025_T60_config() -> dict[str, Any]:
    return {
        "mesh_kind": "structured",
        "mesh_auto_from_wavelength": True,
        "mesh_target_ppw": 15.0,
        "nx": 16,
        "ny": 16,
        "domain_length_x": 17.063911588704,
        "domain_length_y": 17.063911588704,
        "absorbing_layer": True,
        "absorbing_layer_width": 1.528070678994,
        "absorbing_layer_width_wavelengths": 1.5,
        "absorbing_layer_min_cells": 10,
        "absorbing_layer_strength": 6.0,
        "absorbing_layer_polynomial_order": 2.0,
        "source_kind": "internal_gaussian_array",
        "source_array_start_x": 3.531955794352,
        "source_array_start_y": 3.531955794352,
        "source_array_end_x": 13.531955794352,
        "source_array_end_y": 3.531955794352,
        "source_array_num_elements": 21,
        "source_sigma": 0.254678446499,
        "source_direction_x": 0.0,
        "source_direction_y": 1.0,
        "receiver_kind": "internal_gaussian_array",
        "receiver_array_geometry": "line",
        "receiver_array_start_x": 3.531955794352,
        "receiver_array_start_y": 13.531955794352,
        "receiver_array_end_x": 13.531955794352,
        "receiver_array_end_y": 13.531955794352,
        "receiver_array_num_elements": 21,
        "receiver_sigma": 0.254678446499,
        "observation_kind": "vector_array",
        "simulated_material_kind": "healthy",
        "observed_material_kind": "localized_disk_defect",
        "defect_center_x": 8.531955794352,
        "defect_center_y": 8.531955794352,
        "defect_radius": 0.25,
        "contrast_factor": 0.8,
        "theta": 0.25,
        "dt": 0.05,
        "num_steps": 1200,
        "source_frequency": 1.0,
        "every_k": 10,
        "reflection_weight": 1.0,
        "transmission_weight": 1.0,
        "reuse_direct_effective_solver": True,
        "reuse_adjoint_effective_solver": True,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-8,
        "ksp_atol": 1.0e-10,
        "ksp_max_it": 500,
        "visual_output_format": "pvd",
        "export_material_fields": True,
        "export_mesh": True,
        "export_sampling_conditions": True,
    }


def _article_box_except_gamma1_r025_T60_config() -> dict[str, Any]:
    config = _article_face_to_face_r025_T60_config()

    config.update(
        {
            "receiver_array_geometry": "box_except_gamma1",
            "receiver_box_xmin": 3.531955794352,
            "receiver_box_xmax": 13.531955794352,
            "receiver_box_ymin": 3.531955794352,
            "receiver_box_ymax": 13.531955794352,
            "receiver_box_left_num_elements": 21,
            "receiver_box_top_num_elements": 21,
            "receiver_box_right_num_elements": 21,
            "receiver_array_start_x": 3.531955794352,
            "receiver_array_start_y": 13.531955794352,
            "receiver_array_end_x": 13.531955794352,
            "receiver_array_end_y": 13.531955794352,
            "receiver_array_num_elements": 21,
        }
    )

    return config


def _article_L73_N73_face_to_face_r025_T26_no_absorbing_config() -> dict[str, Any]:
    """Article geometry: square [0,73]x[0,73], no absorbing layer.

    Gamma1 = [(30,30),(43,30)] and Gamma2 = [(30,43),(43,43)].
    The defect center is (36.5,36.5), i.e. both the domain center and the
    center of the square region delimited by Gamma1/Gamma2.
    """

    array_num_elements = 21
    array_spacing = 13.0 / float(array_num_elements - 1)
    array_sigma = 0.5 * array_spacing

    return {
        "mesh_kind": "structured",
        "mesh_auto_from_wavelength": False,
        "mesh_target_ppw": 15.0,
        "nx": 73,
        "ny": 73,
        "domain_length_x": 73.0,
        "domain_length_y": 73.0,
        "absorbing_layer": False,
        "absorbing_layer_width": None,

        "source_kind": "internal_gaussian_array",
        "source_array_start_x": 30.0,
        "source_array_start_y": 30.0,
        "source_array_end_x": 43.0,
        "source_array_end_y": 30.0,
        "source_array_num_elements": array_num_elements,
        "source_sigma": array_sigma,
        "source_direction_x": 0.0,
        "source_direction_y": 1.0,

        "receiver_kind": "internal_gaussian_array",
        "receiver_array_geometry": "line",
        "receiver_array_start_x": 30.0,
        "receiver_array_start_y": 43.0,
        "receiver_array_end_x": 43.0,
        "receiver_array_end_y": 43.0,
        "receiver_array_num_elements": array_num_elements,
        "receiver_sigma": array_sigma,

        "observation_kind": "vector_array",
        "simulated_material_kind": "healthy",
        "observed_material_kind": "localized_disk_defect",
        "defect_center_x": 36.5,
        "defect_center_y": 36.5,
        "defect_radius": 0.25,
        "contrast_factor": 0.75,

        "theta": 0.25,
        "dt": 0.05,
        "num_steps": 520,
        "source_frequency": 1.0,
        "every_k": 10,

        "reflection_weight": 1.0,
        "transmission_weight": 1.0,
        "reuse_direct_effective_solver": True,
        "reuse_adjoint_effective_solver": True,

        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-8,
        "ksp_atol": 1.0e-10,
        "ksp_max_it": 500,

        "visual_output_format": "pvd",
        "export_material_fields": True,
        "export_mesh": True,
        "export_sampling_conditions": True,
    }


def _article_L73_gmsh_p1_ppw15_near40_T26_no_absorbing_config() -> dict[str, Any]:
    """Article geometry: [0,73]^2, P1 Gmsh refined mesh, no absorbing layer.

    Global wavelength resolution:
        15 points per minimum wavelength.

    Local defect resolution:
        40 points per minimum wavelength near the defect,
        corresponding to about 20 cells over the defect diameter for r=0.25.

    Gamma1 = [(30,30),(43,30)].
    Gamma2 = [(30,43),(43,43)].
    Defect center = (36.5,36.5).
    """

    array_num_elements = 21
    array_spacing = 13.0 / float(array_num_elements - 1)
    array_sigma = 0.5 * array_spacing

    return {
        "degree": 1,

        "mesh_kind": "gmsh_refined",
        "mesh_auto_from_wavelength": True,
        "mesh_target_ppw": 15.0,
        "mesh_near_target_ppw": 40.0,
        "mesh_refine_defect": True,
        "mesh_refine_defect_radius": 0.75,

        # Positive placeholders required by the generic configuration validator.
        # For gmsh_refined, the effective sizes are mesh_far_h and mesh_near_h
        # computed from the wavelength plan.
        "nx": 1075,
        "ny": 1075,

        "domain_length_x": 73.0,
        "domain_length_y": 73.0,
        "absorbing_layer": False,
        "absorbing_layer_width": None,

        "source_kind": "internal_gaussian_array",
        "source_array_start_x": 30.0,
        "source_array_start_y": 30.0,
        "source_array_end_x": 43.0,
        "source_array_end_y": 30.0,
        "source_array_num_elements": array_num_elements,
        "source_sigma": array_sigma,
        "source_direction_x": 0.0,
        "source_direction_y": 1.0,

        "receiver_kind": "internal_gaussian_array",
        "receiver_array_geometry": "line",
        "receiver_array_start_x": 30.0,
        "receiver_array_start_y": 43.0,
        "receiver_array_end_x": 43.0,
        "receiver_array_end_y": 43.0,
        "receiver_array_num_elements": array_num_elements,
        "receiver_sigma": array_sigma,

        "observation_kind": "vector_array",
        "simulated_material_kind": "healthy",
        "observed_material_kind": "localized_disk_defect",
        "defect_center_x": 36.5,
        "defect_center_y": 36.5,
        "defect_radius": 0.25,
        "contrast_factor": 0.75,

        "theta": 0.25,
        "dt": 0.05,
        "num_steps": 520,
        "source_frequency": 1.0,
        "every_k": 10,

        "reflection_weight": 1.0,
        "transmission_weight": 1.0,
        "reuse_direct_effective_solver": True,
        "reuse_adjoint_effective_solver": True,

        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-8,
        "ksp_atol": 1.0e-10,
        "ksp_max_it": 500,

        "visual_output_format": "pvd",
        "export_material_fields": True,
        "export_mesh": True,
        "export_sampling_conditions": True,
    }


def _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T26_no_absorbing_config() -> dict[str, Any]:
    """Article box-inspection geometry: [0,73]^2, P1 Gmsh refined mesh.

    Gamma1 is the lower side of the square region of interest:
        Gamma1 = [(30,30),(43,30)].

    Gamma2 is the inspection box made of the three remaining sides:
        left  = [(30,30),(30,43)],
        top   = [(30,43),(43,43)],
        right = [(43,30),(43,43)].

    The defect is centered at (36.5,36.5). No absorbing layer is used.
    """

    array_num_elements = 21
    array_spacing = 13.0 / float(array_num_elements - 1)
    array_sigma = 0.5 * array_spacing

    return {
        "degree": 1,

        "mesh_kind": "gmsh_refined",
        "mesh_auto_from_wavelength": True,
        "mesh_target_ppw": 15.0,
        "mesh_near_target_ppw": 40.0,
        "mesh_refine_defect": True,
        "mesh_refine_defect_radius": 0.75,

        # Positive placeholders required by the generic validator.
        # Effective mesh sizes are computed from wavelength-based sizing.
        "nx": 1075,
        "ny": 1075,

        "domain_length_x": 73.0,
        "domain_length_y": 73.0,
        "absorbing_layer": False,
        "absorbing_layer_width": None,

        "source_kind": "internal_gaussian_array",
        "source_array_start_x": 30.0,
        "source_array_start_y": 30.0,
        "source_array_end_x": 43.0,
        "source_array_end_y": 30.0,
        "source_array_num_elements": array_num_elements,
        "source_sigma": array_sigma,
        "source_direction_x": 0.0,
        "source_direction_y": 1.0,

        "receiver_kind": "internal_gaussian_array",
        "receiver_array_geometry": "box_except_gamma1",

        # The real Gamma2 is controlled by these box fields.
        "receiver_box_xmin": 30.0,
        "receiver_box_xmax": 43.0,
        "receiver_box_ymin": 30.0,
        "receiver_box_ymax": 43.0,
        "receiver_box_left_num_elements": array_num_elements,
        "receiver_box_top_num_elements": array_num_elements,
        "receiver_box_right_num_elements": array_num_elements,

        # Compatibility fields: top segment of the box.
        "receiver_array_start_x": 30.0,
        "receiver_array_start_y": 43.0,
        "receiver_array_end_x": 43.0,
        "receiver_array_end_y": 43.0,
        "receiver_array_num_elements": array_num_elements,
        "receiver_sigma": array_sigma,

        "observation_kind": "vector_array",
        "simulated_material_kind": "healthy",
        "observed_material_kind": "localized_disk_defect",
        "defect_center_x": 36.5,
        "defect_center_y": 36.5,
        "defect_radius": 0.25,
        "contrast_factor": 0.75,

        "theta": 0.25,
        "dt": 0.05,
        "num_steps": 520,
        "source_frequency": 1.0,
        "every_k": 10,

        "reflection_weight": 1.0,
        "transmission_weight": 1.0,
        "reuse_direct_effective_solver": True,
        "reuse_adjoint_effective_solver": True,

        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-8,
        "ksp_atol": 1.0e-10,
        "ksp_max_it": 500,

        "visual_output_format": "pvd",
        "export_material_fields": True,
        "export_mesh": True,
        "export_sampling_conditions": True,
    }



def _article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing_config() -> dict[str, Any]:
    """Article-ready L73 line-versus-line configuration.

    The direct/adjoint solver advances 800 steps of size 0.025, hence
    ``T_final = 20``. Receiver arrays are sampled at every solver step and
    volumetric fields are exported every ten steps (period 0.25), with the
    initial and final states explicitly retained by the exporter.
    """

    config = _article_L73_gmsh_p1_ppw15_near40_T26_no_absorbing_config()
    config.update(
        {
            "dt": 0.025,
            "num_steps": 800,
            "every_k": 10,
            "source_num_periods": 1.0,
            "export_initial_state": True,
            "export_final_state": True,
            "article_ready": True,
        }
    )
    return config


def _article_L73_gmsh_p1_ppw15_near40_T20_shear_polarisation_config() -> dict[str, Any]:
    """Symmetry-reversal variant of the article-ready L73 line preset.

    Identical to ``_article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing_config``
    in every respect except the source polarisation, which is rotated from
    ``(0, 1)`` to ``(1, 0)``.

    Why this particular rotation.  Under the mirror ``M : x1 -> 73 - x1`` of the
    inspection set-up, ``(0, 1)`` is mirror-EVEN whereas ``(1, 0)`` is mirror-ODD
    (``M p = -p``).  The excitation therefore lands in the opposite parity class
    rather than merely breaking the parity, and the C66 mirror-symmetry theorem
    *reverses* term by term instead of being destroyed: ``eps11`` and ``eps22``
    become odd and vanish on the axis while ``gamma12`` becomes even and does
    not, so the double zero moves from the C66 kernel to the C11, C22 and C12
    kernels.  The density kernel does not move in either class and serves as a
    control.  This is verified symbolically, ahead of any propagation, by
    ``scripts/revision_wave_motion/verify_mirror_symmetry_symbolic.py`` (checks
    6a-6g).

    Both polarisations are unit vectors, so the source amplitude is unchanged:
    ``source_direction`` is applied without normalisation
    (``elastic_fwi/sources/internal.py``), and any change of norm here would
    confound the comparison with the reference line preset.

    The defect stays at ``(36.5, 36.5)``, hence on the mirror axis: the mesh,
    its refinement disk and every analysis window are identical to the
    reference line preset, and the comparison is like-for-like.
    """

    config = _article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing_config()
    config.update(
        {
            "source_direction_x": 1.0,
            "source_direction_y": 0.0,
        }
    )
    return config


def _article_L73_gmsh_p1_ppw15_near40_T20_defect_off_axis_config() -> dict[str, Any]:
    """Symmetry-breaking variant of the article-ready L73 line preset.

    Identical to ``_article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing_config``
    in every respect except the defect centre, shifted along ``x1`` away from
    the mirror axis ``x1 = 36.5`` by four defect radii (1.0 non-dimensional
    unit), to ``(37.5, 36.5)``.

    Broken hypothesis.  The inspection set-up (domain, source array Gamma1,
    receiver array Gamma2, source polarisation ``(0, 1)``) is invariant under
    the reflection ``M : x1 -> 73 - x1``.  The mirror-symmetry theorem of the
    accompanying article requires the defect centre to lie on that axis,
    i.e. ``defect_center_x = 36.5``, for its proof to force the four
    ``C_66`` kernels to vanish to second order there.  This preset breaks
    exactly that one hypothesis and nothing else: same aperture, same
    polarisation, same contrast factor 0.75, same mesh target, same
    source/dt/T as the reference line preset.  The shift is along ``x1``
    because ``M`` only acts on that coordinate; a shift along ``x2`` alone
    would leave the defect on the (still infinite, ``x2``-independent) axis
    of symmetry and would break nothing.

    Magnitude.  Gamma1 and Gamma2 both span ``x1`` in ``[30, 43]``, a
    half-width of 6.5 non-dimensional units from the axis; the shifted
    centre at 37.5 leaves 3.75 units (15 defect radii) of margin to the
    nearest array edge, so the defect remains well inside the sensing
    aperture.  Four defect radii is also an order of magnitude above the
    mesh-asymmetry floor measured on the on-axis symmetry-reversal campaign
    (barycentre offset 0.88-1.4 per cent of a defect radius): this shift
    cannot be confused with discretisation noise. The sign of the shift is
    immaterial by the mirror symmetry of the set-up itself.

    What this tests.  The C66 sensitivity energy inside the inclusion is
    negligible for every construction while the defect sits on the mirror
    axis.  Whether that near-vanishing is tied to the axis alignment, as the
    theorem's proof implies but does not by itself establish, cannot be
    decided while the defect stays on the axis.  Separating the two
    explanations requires the same scan repeated with the defect off that
    axis, which is what this preset provides.

    Mesh.  Because the refinement disk (``mesh_refine_defect_radius`` =
    0.75) is centred on ``defect_center_x``/``defect_center_y``, moving the
    centre regenerates a genuinely new mesh.  This campaign is therefore NOT
    bit-comparable to the reference line preset via the
    ``standard_reflection_only`` invariance check used between line and box
    (that check relies on Gamma1 being identical, which it is here, but the
    mesh underneath it is not the same mesh as the reference preset's).
    Validation controls must therefore be run on this campaign's own output
    before any number from it is trusted.  The mesh statistics of the two
    configurations are indistinguishable (2 683 952 against 2 684 770 cells,
    discretised defect area within 0.1 per cent).

    Result.  A run takes about 15 h on 32 MPI ranks.  The near vanishing of
    the C66 sensitivity is caused by the defect sitting on the mirror axis:
    the in-target energy fraction rises by two to three orders of magnitude
    once that single hypothesis is broken.
    """

    config = _article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing_config()
    config.update(
        {
            "defect_center_x": 37.5,
        }
    )
    return config


def _article_L73_gmsh_p1_ppw15_near40_T20_decoupled_contrast_config(
    defects_json: str,
) -> dict[str, Any]:
    """Decoupled-contrast variant of the article-ready L73 line preset.

    Identical to ``_article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing_config``
    in every respect except the material assigned inside the inclusion, which
    is read from a single-defect JSON scenario carrying *independent*
    ``rho_factor`` and ``stiffness_factor`` instead of one joint scalar.

    Motivation.  The article defect multiplies ``rho`` and every ``C_ij`` by
    the same factor 0.75, so the Christoffel tensor, homogeneous of degree
    zero under such a joint rescaling, is unchanged and the inclusion is a
    contrast of acoustic impedance at *unchanged wave speed*.  The five
    sensitivity maps are therefore driven by one physically correlated
    perturbation and do not on their own test multiparameter
    identifiability.

    The experiment this preset serves.  Because every ``C_ij`` carries the same
    factor, every phase velocity scales as ``sqrt(stiffness/rho)`` and every
    impedance as ``sqrt(stiffness*rho)``, uniformly in direction and mode.
    The two decoupled scenarios are therefore chosen so that they share one
    impedance contrast and differ only in the sign of the speed contrast:

        published (0.75, 0.75) : speed x1.000, impedance x0.750
        density only (0.75, 1.00) : speed x1.155, impedance x0.866
        stiffness only (1.00, 0.75) : speed x0.866, impedance x0.866

    Any difference between the two decoupled maps is thus attributable to wave
    speed alone, impedance being held fixed, while the published case isolates
    impedance at zero speed contrast.

    Mesh.  ``mesh_refine_from_defects_json`` is set so that the healthy and the
    defective solves both read their refinement disk from the same JSON file;
    otherwise the healthy solve would fall back on ``defect_center_*`` and the
    two would be built by different code paths.  The JSON centre and radius
    repeat the reference line preset values (36.5, 36.5) and 0.25, and
    ``mesh_refine_defect_radius`` stays 0.75, so the refinement disk is
    geometrically identical to the reference line preset and the comparison
    is like-for-like.  ``mesh_near_stiffness_factor`` is deliberately left at
    1.0: the density-only scenario makes waves faster, hence *improves* the
    local resolution, and the stiffness-only scenario lowers it from 40 to
    about 34.6 points per wavelength, still far above the global target of 15.
    Re-sizing the mesh would buy resolution at the cost of comparability with
    the reference line preset, which is the more valuable property here.

    Caveat.  Under ``observed_material_kind='multi_disk_defects'`` the
    ``contrast_factor`` entry is inert: it is validated and recorded in the
    manifest but plays no part in building the material, which comes entirely
    from the JSON file.  Passing ``--contrast-factor`` to this preset is a
    silent no-op.
    """

    config = _article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing_config()
    config.update(
        {
            "observed_material_kind": "multi_disk_defects",
            "defects_json": defects_json,
            "defect_background_factor": 1.0,
            "defect_conflict_policy": "error",
            "mesh_refine_from_defects_json": True,
        }
    )
    return config


def _article_L73_gmsh_p1_ppw15_near40_T20_density_only_config() -> dict[str, Any]:
    """Density-only decoupled contrast: rho x0.75, every C_ij unchanged.

    Wave speed inside the inclusion rises by a factor sqrt(1/0.75) = 1.155.
    See ``_article_L73_gmsh_p1_ppw15_near40_T20_decoupled_contrast_config``.
    """

    return _article_L73_gmsh_p1_ppw15_near40_T20_decoupled_contrast_config(
        "configs/defects/article_L73_single_defect_density_only.json"
    )


def _article_L73_gmsh_p1_ppw15_near40_T20_stiffness_only_config() -> dict[str, Any]:
    """Stiffness-only decoupled contrast: every C_ij x0.75, rho unchanged.

    Wave speed inside the inclusion falls by a factor sqrt(0.75) = 0.866.
    See ``_article_L73_gmsh_p1_ppw15_near40_T20_decoupled_contrast_config``.
    """

    return _article_L73_gmsh_p1_ppw15_near40_T20_decoupled_contrast_config(
        "configs/defects/article_L73_single_defect_stiffness_only.json"
    )


def _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing_config() -> dict[str, Any]:
    """Article-ready L73 box-inspection configuration."""

    config = _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T26_no_absorbing_config()
    config.update(
        {
            "dt": 0.025,
            "num_steps": 800,
            "every_k": 10,
            "source_num_periods": 1.0,
            "export_initial_state": True,
            "export_final_state": True,
            "article_ready": True,
        }
    )
    return config


def _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_shear_polarisation_config() -> dict[str, Any]:
    """Symmetry-reversal variant of the article-ready L73 box preset.

    Identical to
    ``_article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing_config``
    in every respect except the source polarisation, which is rotated from
    ``(0, 1)`` to ``(1, 0)``: the same rotation, and the same argument, as
    ``_article_L73_gmsh_p1_ppw15_near40_T20_shear_polarisation_config`` for the
    ``line`` aperture. See that preset's docstring for the parity argument
    itself (verified symbolically, ahead of any propagation, by
    ``scripts/revision_wave_motion/verify_mirror_symmetry_symbolic.py``,
    checks 6a-6g). The argument is generic in the receiver-array geometry and
    is not re-derived here.

    Purpose.  The parity argument is geometric and applies verbatim to the
    three-sided aperture, which shares the same emitting array and the same
    axis of symmetry.  This preset computes the reversal for that aperture;
    it adds a computation, not a new argument.

    The defect stays at ``(36.5, 36.5)``, hence on the mirror axis, exactly as
    for the ``box`` no-absorbing preset: the mesh, its refinement disk and
    every analysis window are identical, so the comparison against the
    ``box`` T20 reference preset remains like-for-like. Gamma1 (the emitting
    array, unchanged by ``box``) and the mirror axis are shared with ``line``,
    which is exactly the geometric invariance the argument relies on.
    """

    config = _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing_config()
    config.update(
        {
            "source_direction_x": 1.0,
            "source_direction_y": 0.0,
        }
    )
    return config


def _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_sparse_config() -> dict[str, Any]:
    """Constant-sensor-budget variant of the article-ready L73 box preset.

    Identical to ``_article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing_config``
    except that each of the three sides carries eight elements instead of
    twenty-one.  Corners are shared between adjacent sides, so the total is
    ``3 * 8 - 2 = 22`` receivers against ``3 * 21 - 2 = 61``.

    Motivation.  Every line-versus-box comparison changes the aperture shape
    and the receiver count together (21 against 61), so none of them can
    attribute the box gain to one factor rather than the other.  Running the
    three-sided aperture at a budget comparable to the facing one separates
    the two, and it answers the practical question directly: given a fixed
    number of transducers, is it better to concentrate them facing the defect
    or to spread them over three sides?

    Why 22 and not 21.  With shared corners the reachable totals are ``3n - 2``,
    that is 19 for seven elements per side and 22 for eight; 21 is not
    reachable with a uniform per-side count.  Twenty-two is the closest, and it
    errs marginally in favour of the three-sided aperture, so a loss against the
    facing 21 would be conclusive a fortiori.

    Known and deliberate consequence: spatial under-sampling.  Twenty-two
    receivers spread over a perimeter of 39 give a pitch of 1.857 against 0.650
    for both reference apertures, that is 1.08 points per wavelength at the
    nominal source frequency against 3.08, below the Shannon threshold of
    two.  The element pitch also grows from two receiver standard deviations to
    5.7, so the array stops being a quasi-continuous aperture and becomes a set
    of isolated sensors.  This is not an artefact to be corrected: it is the
    physical price of the economy being tested.
    """

    config = _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing_config()
    config.update(
        {
            "receiver_box_left_num_elements": 8,
            "receiver_box_top_num_elements": 8,
            "receiver_box_right_num_elements": 8,
        }
    )
    return config


def _article_L120_gmsh_p1_ppw15_near40_multidefect_T42_no_absorbing_config() -> dict[str, Any]:
    """Five-defect line-versus-line article geometry on ``[0,120]^2``.

    Gamma1 is the lower line of the region of interest:
        [(50,50),(70,50)].

    Gamma2 is the upper line:
        [(50,70),(70,70)].

    The healthy and defective forward calculations use the same five Gmsh
    refinement disks read from the defect JSON file.  The local
    mesh size accounts for the softest stiffness factor, 0.5, at unchanged
    density, so that the minimum local resolution is 40 points per wavelength.
    """

    array_num_elements = 32
    array_spacing = 20.0 / float(array_num_elements - 1)
    array_sigma = 0.5 * array_spacing
    defects_json = "configs/defects/article_L120_multidefect_5_elastic_only.json"

    return {
        "degree": 1,

        "mesh_kind": "gmsh_refined",
        "mesh_auto_from_wavelength": True,
        "mesh_target_ppw": 15.0,
        "mesh_near_target_ppw": 40.0,
        "mesh_near_stiffness_factor": 0.5,
        "mesh_refine_defect": True,
        "mesh_refine_from_defects_json": True,
        "mesh_refine_defect_radius": None,

        # Positive placeholders required by the generic configuration
        # validator.  The Gmsh sizes are computed from the wavelength plan.
        "nx": 1767,
        "ny": 1767,

        "domain_length_x": 120.0,
        "domain_length_y": 120.0,
        "absorbing_layer": False,
        "absorbing_layer_width": None,

        "source_kind": "internal_gaussian_array",
        "source_array_start_x": 50.0,
        "source_array_start_y": 50.0,
        "source_array_end_x": 70.0,
        "source_array_end_y": 50.0,
        "source_array_num_elements": array_num_elements,
        "source_sigma": array_sigma,
        "source_direction_x": 0.0,
        "source_direction_y": 1.0,

        "receiver_kind": "internal_gaussian_array",
        "receiver_array_geometry": "line",
        "receiver_array_start_x": 50.0,
        "receiver_array_start_y": 70.0,
        "receiver_array_end_x": 70.0,
        "receiver_array_end_y": 70.0,
        "receiver_array_num_elements": array_num_elements,
        "receiver_sigma": array_sigma,

        "observation_kind": "vector_array",
        "simulated_material_kind": "healthy",
        "observed_material_kind": "multi_disk_defects",
        "defects_json": defects_json,
        "defect_background_factor": 1.0,
        "defect_conflict_policy": "error",

        "theta": 0.25,
        "dt": 0.05,
        "num_steps": 840,
        "source_frequency": 1.0,
        "every_k": 10,

        "reflection_weight": 1.0,
        "transmission_weight": 1.0,
        "reuse_direct_effective_solver": True,
        "reuse_adjoint_effective_solver": True,

        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-8,
        "ksp_atol": 1.0e-10,
        "ksp_max_it": 500,

        "visual_output_format": "pvd",
        "export_material_fields": True,
        "export_mesh": True,
        "export_sampling_conditions": True,
    }


def _article_L120_gmsh_p1_ppw15_near40_multidefect_box_except_gamma1_T42_no_absorbing_config() -> dict[str, Any]:
    """Five-defect box-inspection variant of the L120 article geometry."""

    config = _article_L120_gmsh_p1_ppw15_near40_multidefect_T42_no_absorbing_config()
    array_num_elements = int(config["receiver_array_num_elements"])

    config.update(
        {
            "receiver_array_geometry": "box_except_gamma1",
            "receiver_box_xmin": 50.0,
            "receiver_box_xmax": 70.0,
            "receiver_box_ymin": 50.0,
            "receiver_box_ymax": 70.0,
            "receiver_box_left_num_elements": array_num_elements,
            "receiver_box_top_num_elements": array_num_elements,
            "receiver_box_right_num_elements": array_num_elements,
            # Compatibility line on the top side.  The actual
            # Gamma2 observation is the left+top+right polyline.
            "receiver_array_start_x": 50.0,
            "receiver_array_start_y": 70.0,
            "receiver_array_end_x": 70.0,
            "receiver_array_end_y": 70.0,
        }
    )

    return config


def _config_for_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.preset == "article_face_to_face_r025_T60":
        config = _article_face_to_face_r025_T60_config()
    elif args.preset == "article_box_except_gamma1_r025_T60":
        config = _article_box_except_gamma1_r025_T60_config()
    elif args.preset == "article_L73_N73_face_to_face_r025_T26_no_absorbing":
        config = _article_L73_N73_face_to_face_r025_T26_no_absorbing_config()
    elif args.preset == "article_L73_gmsh_p1_ppw15_near40_T26_no_absorbing":
        config = _article_L73_gmsh_p1_ppw15_near40_T26_no_absorbing_config()
    elif args.preset == "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T26_no_absorbing":
        config = _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T26_no_absorbing_config()
    elif args.preset == "article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing":
        config = _article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing_config()
    elif args.preset == "article_L73_gmsh_p1_ppw15_near40_T20_shear_polarisation":
        config = _article_L73_gmsh_p1_ppw15_near40_T20_shear_polarisation_config()
    elif args.preset == "article_L73_gmsh_p1_ppw15_near40_T20_defect_off_axis":
        config = _article_L73_gmsh_p1_ppw15_near40_T20_defect_off_axis_config()
    elif args.preset == "article_L73_gmsh_p1_ppw15_near40_T20_density_only":
        config = _article_L73_gmsh_p1_ppw15_near40_T20_density_only_config()
    elif args.preset == "article_L73_gmsh_p1_ppw15_near40_T20_stiffness_only":
        config = _article_L73_gmsh_p1_ppw15_near40_T20_stiffness_only_config()
    elif args.preset == "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing":
        config = _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing_config()
    elif args.preset == "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_shear_polarisation":
        config = _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_shear_polarisation_config()
    elif args.preset == "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_sparse":
        config = _article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_sparse_config()
    elif args.preset == "article_L120_gmsh_p1_ppw15_near40_multidefect_T42_no_absorbing":
        config = _article_L120_gmsh_p1_ppw15_near40_multidefect_T42_no_absorbing_config()
    elif args.preset == "article_L120_gmsh_p1_ppw15_near40_multidefect_box_except_gamma1_T42_no_absorbing":
        config = _article_L120_gmsh_p1_ppw15_near40_multidefect_box_except_gamma1_T42_no_absorbing_config()
    else:
        raise ValueError(
            "Unsupported preset for this standalone campaign script: "
            f"{args.preset!r}. Use 'article_face_to_face_r025_T60', "
            "'article_box_except_gamma1_r025_T60', "
            "'article_L73_N73_face_to_face_r025_T26_no_absorbing', "
            "'article_L73_gmsh_p1_ppw15_near40_T26_no_absorbing', or "
            "'article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T26_no_absorbing', "
            "'article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing', "
            "'article_L73_gmsh_p1_ppw15_near40_T20_shear_polarisation', "
            "'article_L73_gmsh_p1_ppw15_near40_T20_defect_off_axis', "
            "'article_L73_gmsh_p1_ppw15_near40_T20_density_only', "
            "'article_L73_gmsh_p1_ppw15_near40_T20_stiffness_only', "
            "'article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing', "
            "'article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_shear_polarisation', "
            "'article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_sparse', "
            "'article_L120_gmsh_p1_ppw15_near40_multidefect_T42_no_absorbing', or "
            "'article_L120_gmsh_p1_ppw15_near40_multidefect_box_except_gamma1_T42_no_absorbing'."
        )

    if args.quick:
        config = dict(config)
        config["num_steps"] = 10
        config["every_k"] = 1

    config = dict(config)

    # All campaign wave problems share one physical-experiment mesh.  The
    # mesh is built from the observed/defective configuration so that defect
    # refinement is present even for wave solves performed with healthy
    # material coefficients.
    config.setdefault("common_mesh_direct_pair", True)

    if args.dt is not None:
        config["dt"] = float(args.dt)
    if args.num_steps is not None:
        config["num_steps"] = int(args.num_steps)
    if args.every_k is not None:
        config["every_k"] = int(args.every_k)
    if args.contrast_factor is not None:
        config["contrast_factor"] = float(args.contrast_factor)
    if args.source_num_periods is not None:
        config["source_num_periods"] = float(args.source_num_periods)
    if args.export_initial_state is not None:
        config["export_initial_state"] = bool(args.export_initial_state)
    if args.export_final_state is not None:
        config["export_final_state"] = bool(args.export_final_state)

    config["visual_output_format"] = args.visual_output_format
    config["gradient_export_mode"] = args.gradient_export_mode
    config["ksp_type"] = args.ksp_type
    config["pc_type"] = args.pc_type
    config["ksp_rtol"] = args.ksp_rtol
    config["ksp_atol"] = args.ksp_atol
    config["ksp_max_it"] = args.ksp_max_it

    if args.no_material_fields:
        config["export_material_fields"] = False
    if args.no_mesh:
        config["export_mesh"] = False
    if args.no_sampling:
        config["export_sampling_conditions"] = False

    return config


def _vtk_pvd_paths(payload: Mapping[str, Any]) -> dict[str, str]:
    vtk = payload.get("vtk", {})
    if not isinstance(vtk, Mapping):
        return {}

    out: dict[str, str] = {}
    for field_name, field_payload in vtk.items():
        if isinstance(field_payload, Mapping) and field_payload.get("pvd"):
            out[str(field_name)] = str(field_payload["pvd"])
    return out


def _validate_gradient_combinations(
    manifest: Mapping[str, Any],
    *,
    combinations_key: str = "gradient_combinations",
) -> dict[str, Any]:
    exports = manifest.get("exports", {})
    if not isinstance(exports, Mapping):
        return {
            "status": "error",
            "reason": "missing manifest.exports",
            "num_expected_images": sum(len(_required_fields_for_combination(c)) for c in REQUIRED_COMBINATIONS),
            "num_found_images": 0,
        }

    combinations = exports.get(combinations_key, {})
    if not isinstance(combinations, Mapping):
        return {
            "status": "error",
            "reason": f"missing manifest.exports.{combinations_key}",
            "num_expected_images": sum(len(_required_fields_for_combination(c)) for c in REQUIRED_COMBINATIONS),
            "num_found_images": 0,
        }

    missing: list[dict[str, str]] = []
    found: dict[str, dict[str, str]] = {}

    for combination in REQUIRED_COMBINATIONS:
        payload = combinations.get(combination)
        if not isinstance(payload, Mapping):
            for field in _required_fields_for_combination(combination):
                missing.append({"combination": combination, "field": field})
            continue

        paths = _vtk_pvd_paths(payload)
        found[combination] = paths

        for field in _required_fields_for_combination(combination):
            path = paths.get(field)
            if not path:
                missing.append({"combination": combination, "field": field})
                continue
            if not Path(path).exists():
                missing.append({"combination": combination, "field": field})

    num_expected = sum(len(_required_fields_for_combination(c)) for c in REQUIRED_COMBINATIONS)
    num_missing = len(missing)
    num_found = num_expected - num_missing

    return {
        "status": "ok" if num_missing == 0 else "error",
        "combinations_key": str(combinations_key),
        "num_expected_images": num_expected,
        "num_found_images": num_found,
        "num_missing_images": num_missing,
        "required_combinations": list(REQUIRED_COMBINATIONS),
        "required_fields_by_combination": {
            c: list(_required_fields_for_combination(c)) for c in REQUIRED_COMBINATIONS
        },
        "found": found,
        "missing": missing,
    }


def _validate_gradient_spatial_representation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Require every exported primary map to use DG0/L2 cell-average density."""
    exports = manifest.get("exports", {})
    if not isinstance(exports, Mapping):
        return {"status": "error", "reason": "missing manifest.exports", "rows": []}

    rows: list[dict[str, Any]] = []
    for combinations_key in ("gradient_combinations", "gradient_combinations_raw_signed"):
        combinations = exports.get(combinations_key)
        if combinations is None:
            continue
        if not isinstance(combinations, Mapping):
            rows.append({
                "combinations_key": combinations_key,
                "status": "error",
                "reason": "section_not_mapping",
            })
            continue
        for combination in REQUIRED_COMBINATIONS:
            payload = combinations.get(combination)
            metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
            representation = metadata.get("spatial_representation") if isinstance(metadata, Mapping) else None
            normalized = metadata.get("cell_measure_normalization") if isinstance(metadata, Mapping) else None
            ok = (
                representation == "DG0_L2_cell_average_density"
                and normalized is True
            )
            rows.append({
                "combinations_key": combinations_key,
                "combination": combination,
                "spatial_representation": representation,
                "cell_measure_normalization": normalized,
                "status": "ok" if ok else "error",
            })

    required_rows = len(REQUIRED_COMBINATIONS)
    article_rows = [row for row in rows if row.get("combinations_key") == "gradient_combinations"]
    status = "ok" if (
        len(article_rows) == required_rows
        and rows
        and all(row.get("status") == "ok" for row in rows)
    ) else "error"
    return {
        "status": status,
        "expected_spatial_representation": "DG0_L2_cell_average_density",
        "cell_measure_normalization_required": True,
        "num_rows": len(rows),
        "rows": rows,
    }


def run_campaign(args: argparse.Namespace) -> dict[str, Any] | None:
    comm = MPI.COMM_WORLD
    output_dir = Path(args.output_dir)

    if comm.rank == 0 and args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    config = _config_for_args(args)

    t0 = time.perf_counter()
    inverse_manifest = export_inverse_time_series_xdmf(
        output_dir=output_dir / "inverse",
        comm=comm,
        **config,
    )
    comm.Barrier()
    elapsed = time.perf_counter() - t0

    if comm.rank != 0:
        return None

    validation = _validate_gradient_combinations(inverse_manifest)
    raw_signed_validation = None
    if config.get("gradient_export_mode") == "both":
        raw_signed_validation = _validate_gradient_combinations(
            inverse_manifest,
            combinations_key="gradient_combinations_raw_signed",
        )

    gradient_spatial_contract = _validate_gradient_spatial_representation(inverse_manifest)

    direct_mesh_contract = inverse_manifest.get("direct_mesh_contract", {})
    common_mesh_validation_ok = (
        not bool(config.get("common_mesh_direct_pair", True))
        or direct_mesh_contract.get("status") == "ok"
    )

    overall_status_ok = (
        validation["status"] == "ok"
        and (
            raw_signed_validation is None
            or raw_signed_validation["status"] == "ok"
        )
        and common_mesh_validation_ok
        and gradient_spatial_contract["status"] == "ok"
    )

    campaign_manifest = {
        "kind": "gradient_combination_campaign",
        "status": "ok" if overall_status_ok else "error",
        "description": (
            "Single optimized campaign for first-gradient image combinations: "
            "u_ref x lambda_Gamma1, u_ref x lambda_Gamma2, "
            "u_ref x lambda_Gamma1+Gamma2, and W_Gamma1 x lambda_Gamma2."
        ),
        "mpi_size": comm.size,
        "elapsed_seconds": float(elapsed),
        "preset": "quick" if args.quick else args.preset,
        "output_dir": str(output_dir),
        "inverse_output_dir": str(output_dir / "inverse"),
        "inverse_manifest": str(
            output_dir
            / "inverse"
            / (
                f"inverse_time_series_manifest_rank{comm.rank:04d}of{comm.size:04d}.json"
                if comm.size > 1
                else "inverse_time_series_manifest.json"
            )
        ),
        "solve_count": {
            "direct_observed_defective": 1,
            "direct_simulated_reference": 1,
            "adjoint_full_gamma1_plus_gamma2": 1,
            "adjoint_reflection_gamma1": 1,
            "adjoint_transmission_gamma2": 1,
            "total_wave_solves": 5,
        },
        "expected_images": {
            "num_combinations": len(REQUIRED_COMBINATIONS),
            "num_gradient_fields_per_gradient_combination": len(REQUIRED_GRADIENT_FIELDS),
            "num_kernel_fields_for_residual_interaction": len(REQUIRED_KERNEL_FIELDS),
            "num_primary_images": sum(
                len(_required_fields_for_combination(c)) for c in REQUIRED_COMBINATIONS
            ),
        },
        "validation": validation,
        "raw_signed_validation": raw_signed_validation,
        "gradient_spatial_contract": gradient_spatial_contract,
        "direct_mesh_contract": direct_mesh_contract,
        "common_mesh_validation_ok": bool(common_mesh_validation_ok),
        "article_ready_exports": inverse_manifest.get("article_ready_exports"),
        "temporal_sampling": inverse_manifest.get("temporal_sampling"),
        "mesh_discretization": inverse_manifest.get("mesh_discretization"),
        "config": config,
    }

    try:
        from elastic_fwi.tracking.tracked_manifest import enrich_manifest_with_tracking

        campaign_manifest = enrich_manifest_with_tracking(
            campaign_manifest,
            output_dir=output_dir,
            args=vars(args),
            comm=comm,
            elapsed_seconds=float(elapsed),
        )
    except Exception as exc:
        campaign_manifest["tracking_error"] = repr(exc)

    manifest_path = output_dir / "gradient_combination_campaign_manifest.json"
    manifest_path.write_text(
        json.dumps(campaign_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(campaign_manifest, indent=2, sort_keys=True), flush=True)
    return campaign_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one optimized inverse export and stop at the 20 primary "
            "material-gradient maps and residual-interaction kernel maps for the four aperture combinations."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preset",
        choices=(
            "article_face_to_face_r025_T60",
            "article_box_except_gamma1_r025_T60",
            "article_L73_N73_face_to_face_r025_T26_no_absorbing",
            "article_L73_gmsh_p1_ppw15_near40_T26_no_absorbing",
            "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T26_no_absorbing",
            "article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing",
            "article_L73_gmsh_p1_ppw15_near40_T20_shear_polarisation",
            "article_L73_gmsh_p1_ppw15_near40_T20_defect_off_axis",
            "article_L73_gmsh_p1_ppw15_near40_T20_density_only",
            "article_L73_gmsh_p1_ppw15_near40_T20_stiffness_only",
            "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing",
            "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_shear_polarisation",
            "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_sparse",
            "article_L120_gmsh_p1_ppw15_near40_multidefect_T42_no_absorbing",
            "article_L120_gmsh_p1_ppw15_near40_multidefect_box_except_gamma1_T42_no_absorbing",
        ),
        default="article_face_to_face_r025_T60",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--every-k", type=int, default=None)
    parser.add_argument(
        "--contrast-factor",
        type=float,
        default=None,
        help=(
            "Override the defect contrast factor of the selected preset. "
            "When omitted, the preset value is used unchanged."
        ),
    )
    parser.add_argument("--source-num-periods", type=float, default=None)
    parser.add_argument(
        "--export-initial-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly include the physical initial state in volumetric exports.",
    )
    parser.add_argument(
        "--export-final-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Always include the physical final state in volumetric exports.",
    )
    parser.add_argument(
        "--visual-output-format",
        choices=("pvd", "xdmf", "both"),
        default="pvd",
    )
    parser.add_argument(
        "--gradient-export-mode",
        choices=("article_safe", "raw_signed", "both"),
        default="article_safe",
        help=(
            "article_safe: normalized absolute-value maps; "
            "raw_signed: signed non-normalized gradient densities; "
            "both: export both families."
        ),
    )
    parser.add_argument("--ksp-type", type=str, default="cg")
    parser.add_argument("--pc-type", type=str, default="hypre")
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-8)
    parser.add_argument("--ksp-atol", type=float, default=1.0e-10)
    parser.add_argument("--ksp-max-it", type=int, default=500)
    parser.add_argument("--no-material-fields", action="store_true")
    parser.add_argument("--no-mesh", action="store_true")
    parser.add_argument("--no-sampling", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_campaign(args)


if __name__ == "__main__":
    main()
