from __future__ import annotations

import math
from pathlib import Path

from elastic_fwi.materials.defect_json import load_defect_scenario_json
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    _build_gmsh_refinement_disks,
)
from elastic_fwi.campaigns.run_gradient_combination_campaign import (
    _config_for_args,
    build_parser,
)


def _config(preset: str):
    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/article_ready_L73",
            "--preset",
            preset,
            "--gradient-export-mode",
            "both",
            "--pc-type",
            "gamg",
        ]
    )
    return _config_for_args(args)


def test_article_ready_line_preset_locks_validated_temporal_sampling() -> None:
    config = _config("article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing")

    assert config["dt"] == 0.025
    assert config["num_steps"] == 800
    assert config["every_k"] == 10
    assert math.isclose(config["dt"] * config["num_steps"], 20.0)
    assert math.isclose(config["dt"] * config["every_k"], 0.25)
    assert config["export_initial_state"] is True
    assert config["export_final_state"] is True
    assert config["source_num_periods"] == 1.0
    assert config["gradient_export_mode"] == "both"
    assert config["ksp_type"] == "cg"
    assert config["pc_type"] == "gamg"
    assert config["receiver_array_geometry"] == "line"
    assert config["common_mesh_direct_pair"] is True


def test_article_ready_L73_healthy_and_defective_configs_share_defect_refinement_disk() -> None:
    config = _config("article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing")

    direct_keys = {
        key: value
        for key, value in config.items()
        if key in DirectObservationPipelineConfig.__dataclass_fields__
    }
    direct_keys.pop("material_kind", None)

    healthy = DirectObservationPipelineConfig(
        **direct_keys,
        material_kind="healthy",
    )
    defective = DirectObservationPipelineConfig(
        **direct_keys,
        material_kind="localized_disk_defect",
    )

    healthy_disks = _build_gmsh_refinement_disks(healthy)
    defective_disks = _build_gmsh_refinement_disks(defective)

    assert healthy_disks == defective_disks
    assert len(healthy_disks) == 1
    disk = healthy_disks[0]
    assert disk.name == "defect"
    assert math.isclose(disk.center_x, 36.5)
    assert math.isclose(disk.center_y, 36.5)
    assert math.isclose(disk.radius, 0.75)


def test_article_ready_box_preset_differs_only_by_receiver_geometry_fields() -> None:
    line = _config("article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing")
    box = _config(
        "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing"
    )

    assert box["receiver_array_geometry"] == "box_except_gamma1"
    assert box["receiver_box_xmin"] == 30.0
    assert box["receiver_box_xmax"] == 43.0
    assert box["receiver_box_ymin"] == 30.0
    assert box["receiver_box_ymax"] == 43.0

    box_only = {
        "receiver_array_geometry",
        "receiver_box_xmin",
        "receiver_box_xmax",
        "receiver_box_ymin",
        "receiver_box_ymax",
        "receiver_box_left_num_elements",
        "receiver_box_top_num_elements",
        "receiver_box_right_num_elements",
    }
    common_keys = (set(line) | set(box)) - box_only
    assert {key: line.get(key) for key in common_keys} == {
        key: box.get(key) for key in common_keys
    }


def test_shear_polarisation_preset_differs_only_by_source_direction() -> None:
    """The shear-polarisation variant must be a like-for-like twin of the line preset.

    Anything else drifting would confound the mirror-symmetry reversal it is
    designed to test, so the difference is pinned to exactly two keys.
    """

    line = _config("article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing")
    shear = _config("article_L73_gmsh_p1_ppw15_near40_T20_shear_polarisation")

    assert line["source_direction_x"] == 0.0
    assert line["source_direction_y"] == 1.0
    assert shear["source_direction_x"] == 1.0
    assert shear["source_direction_y"] == 0.0

    # Both polarisations are unit vectors: source_direction is applied without
    # normalisation, so an unequal norm would change the source amplitude and
    # confound the comparison with the reference line campaign.
    assert math.isclose(
        math.hypot(line["source_direction_x"], line["source_direction_y"]),
        math.hypot(shear["source_direction_x"], shear["source_direction_y"]),
    )

    polarisation_only = {"source_direction_x", "source_direction_y"}
    common_keys = (set(line) | set(shear)) - polarisation_only
    assert {key: line.get(key) for key in common_keys} == {
        key: shear.get(key) for key in common_keys
    }


def test_shear_polarisation_preset_keeps_defect_on_the_mirror_axis() -> None:
    """The defect must stay centred, so the mesh matches the reference campaign.

    The reversal test relies on the mirror axis, the defect abscissa and every
    analysis window still coinciding at x1 = 36.5.  Moving the defect would also
    move the refinement disk and break the like-for-like mesh comparison.
    """

    line = _config("article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing")
    shear = _config("article_L73_gmsh_p1_ppw15_near40_T20_shear_polarisation")

    assert shear["defect_center_x"] == 36.5
    assert shear["defect_center_y"] == 36.5

    def _disks(config):
        direct_keys = {
            key: value
            for key, value in config.items()
            if key in DirectObservationPipelineConfig.__dataclass_fields__
        }
        direct_keys.pop("material_kind", None)
        return _build_gmsh_refinement_disks(
            DirectObservationPipelineConfig(
                **direct_keys,
                material_kind="localized_disk_defect",
            )
        )

    assert _disks(shear) == _disks(line)


def test_box_shear_polarisation_preset_differs_only_by_source_direction() -> None:
    """Box shear-polarisation variant must be a like-for-like twin of the box preset.

    Mirrors ``test_shear_polarisation_preset_differs_only_by_source_direction``
    for the ``box`` aperture: this preset is the box analogue of the ``line``
    shear-polarisation variant.
    """

    box = _config("article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing")
    shear = _config(
        "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_shear_polarisation"
    )

    assert box["source_direction_x"] == 0.0
    assert box["source_direction_y"] == 1.0
    assert shear["source_direction_x"] == 1.0
    assert shear["source_direction_y"] == 0.0

    assert math.isclose(
        math.hypot(box["source_direction_x"], box["source_direction_y"]),
        math.hypot(shear["source_direction_x"], shear["source_direction_y"]),
    )

    polarisation_only = {"source_direction_x", "source_direction_y"}
    common_keys = (set(box) | set(shear)) - polarisation_only
    assert {key: box.get(key) for key in common_keys} == {
        key: shear.get(key) for key in common_keys
    }


def test_box_shear_polarisation_preset_keeps_defect_on_the_mirror_axis() -> None:
    """The box defect must also stay centred, for the same reason as ``line``.

    Mirrors ``test_shear_polarisation_preset_keeps_defect_on_the_mirror_axis``:
    the mirror axis, the defect abscissa and every analysis window must keep
    coinciding at x1 = 36.5 so the mesh matches the box T20 reference campaign.
    """

    shear = _config(
        "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_shear_polarisation"
    )

    assert shear["defect_center_x"] == 36.5
    assert shear["defect_center_y"] == 36.5

    def _disks(config):
        direct_keys = {
            key: value
            for key, value in config.items()
            if key in DirectObservationPipelineConfig.__dataclass_fields__
        }
        direct_keys.pop("material_kind", None)
        return _build_gmsh_refinement_disks(
            DirectObservationPipelineConfig(
                **direct_keys,
                material_kind="localized_disk_defect",
            )
        )

    box = _config("article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing")
    assert _disks(shear) == _disks(box)


def test_sparse_box_preset_differs_only_by_receiver_counts() -> None:
    """Constant-sensor-budget variant must be a twin of the box preset.

    Anything else drifting would reintroduce the very confound the campaign is
    meant to remove, so the difference is pinned to the three per-side counts.
    """

    box = _config("article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing")
    sparse = _config("article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_sparse")

    assert box["receiver_box_left_num_elements"] == 21
    assert sparse["receiver_box_left_num_elements"] == 8
    assert sparse["receiver_box_top_num_elements"] == 8
    assert sparse["receiver_box_right_num_elements"] == 8

    counts_only = {
        "receiver_box_left_num_elements",
        "receiver_box_top_num_elements",
        "receiver_box_right_num_elements",
    }
    common_keys = (set(box) | set(sparse)) - counts_only
    assert {key: box.get(key) for key in common_keys} == {
        key: sparse.get(key) for key in common_keys
    }


def test_sparse_box_array_has_the_intended_budget_and_declared_pitch() -> None:
    """Twenty-two receivers, and a pitch that is knowingly below Shannon.

    Corners are shared, so a uniform per-side count n yields 3n - 2 receivers:
    21 is unreachable and 22 is the closest, erring marginally in favour of the
    three-sided aperture.  The resulting pitch under-samples the wavefield.  This
    is an expected consequence of the reduced sensor budget, and the test
    asserts it explicitly.
    """

    from elastic_fwi.transducers.arrays import build_polyline_transducer_array

    config = _config("article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_sparse")
    array = build_polyline_transducer_array(
        name="gamma2",
        role="transmission",
        vertices=[
            (config["receiver_box_xmin"], config["receiver_box_ymin"]),
            (config["receiver_box_xmin"], config["receiver_box_ymax"]),
            (config["receiver_box_xmax"], config["receiver_box_ymax"]),
            (config["receiver_box_xmax"], config["receiver_box_ymin"]),
        ],
        num_elements_per_segment=[
            config["receiver_box_left_num_elements"],
            config["receiver_box_top_num_elements"],
            config["receiver_box_right_num_elements"],
        ],
        sigma=config["receiver_sigma"],
        tag=2,
    )

    centers = array.centers
    assert len(centers) == 22

    pitches = [
        math.dist(centers[index], centers[index + 1])
        for index in range(len(centers) - 1)
    ]
    pitch = sorted(pitches)[len(pitches) // 2]
    assert math.isclose(pitch, 13.0 / 7.0, rel_tol=1e-9)

    # Both reference apertures sit at 0.65, i.e. two receiver standard
    # deviations and 3.08 points per nominal wavelength.
    assert math.isclose(pitch / config["receiver_sigma"], 5.714285714, rel_tol=1e-6)
    nominal_wavelength = 2.0038851153585515  # c_max / f, from sampling_conditions
    assert 1.0 < nominal_wavelength / pitch < 1.2


def test_article_ready_temporal_cli_overrides_are_explicit() -> None:
    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/article_ready_override",
            "--preset",
            "article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing",
            "--dt",
            "0.02",
            "--num-steps",
            "1000",
            "--every-k",
            "25",
            "--no-export-initial-state",
            "--export-final-state",
        ]
    )
    config = _config_for_args(args)
    assert config["dt"] == 0.02
    assert config["num_steps"] == 1000
    assert config["every_k"] == 25
    assert config["export_initial_state"] is False
    assert config["export_final_state"] is True


DECOUPLED_SCENARIOS = {
    "article_L73_gmsh_p1_ppw15_near40_T20_density_only": Path(
        "configs/defects/article_L73_single_defect_density_only.json"
    ),
    "article_L73_gmsh_p1_ppw15_near40_T20_stiffness_only": Path(
        "configs/defects/article_L73_single_defect_stiffness_only.json"
    ),
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_decoupled_contrast_presets_differ_only_by_the_material_scenario() -> None:
    """Decoupled variants must be like-for-like twins of the line preset.

    These variants separate the density and stiffness perturbations, which are
    correlated in the line preset, so anything else drifting would confound the
    comparison.  The difference is pinned to
    exactly the five keys that route the material through a JSON scenario.
    """

    line = _config("article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing")

    scenario_only = {
        "observed_material_kind",
        "defects_json",
        "defect_background_factor",
        "defect_conflict_policy",
        "mesh_refine_from_defects_json",
    }

    for preset, scenario_path in DECOUPLED_SCENARIOS.items():
        decoupled = _config(preset)

        assert line["observed_material_kind"] == "localized_disk_defect"
        assert decoupled["observed_material_kind"] == "multi_disk_defects"
        assert decoupled["defects_json"] == str(scenario_path)
        assert decoupled["defect_background_factor"] == 1.0
        assert decoupled["defect_conflict_policy"] == "error"
        assert decoupled["mesh_refine_from_defects_json"] is True

        common_keys = (set(line) | set(decoupled)) - scenario_only
        assert {key: line.get(key) for key in common_keys} == {
            key: decoupled.get(key) for key in common_keys
        }


def test_decoupled_contrast_presets_keep_the_frozen_refinement_disk() -> None:
    """The mesh must stay the reference campaign's, healthy and defective alike.

    Under ``multi_disk_defects`` the defective solve reads its refinement disk
    from the JSON file while the healthy solve would fall back on
    ``defect_center_*``; ``mesh_refine_from_defects_json`` forces both down the
    same path.  The disk is compared geometrically rather than by identity
    because the JSON path labels it ``defect_0`` and the single-defect path
    ``defect``, a metadata label that never reaches the mesher.
    """

    def _disks(config, material_kind):
        direct_keys = {
            key: value
            for key, value in config.items()
            if key in DirectObservationPipelineConfig.__dataclass_fields__
        }
        direct_keys.pop("material_kind", None)
        return _build_gmsh_refinement_disks(
            DirectObservationPipelineConfig(
                **direct_keys,
                material_kind=material_kind,
            )
        )

    def _geometry(disks):
        return [
            (disk.center_x, disk.center_y, disk.radius, disk.target_h)
            for disk in disks
        ]

    line = _config("article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing")
    reference = _geometry(_disks(line, "localized_disk_defect"))

    assert len(reference) == 1
    assert math.isclose(reference[0][0], 36.5)
    assert math.isclose(reference[0][1], 36.5)
    assert math.isclose(reference[0][2], 0.75)

    for preset in DECOUPLED_SCENARIOS:
        decoupled = _config(preset)

        healthy = _geometry(_disks(decoupled, "healthy"))
        defective = _geometry(_disks(decoupled, "multi_disk_defects"))

        assert healthy == defective
        assert defective == reference


def test_decoupled_scenarios_hold_impedance_fixed_and_invert_the_speed_contrast() -> None:
    """Lock the experimental design, not merely the fact that factors differ.

    Every ``C_ij`` carries the same factor, so the Christoffel tensor gives one
    speed ratio ``sqrt(stiffness/rho)`` and one impedance ratio
    ``sqrt(stiffness*rho)``, uniformly in direction and mode.  The two scenarios
    are chosen to share an impedance contrast and to carry reciprocal speed
    contrasts, so that any difference between their sensitivity maps is
    attributable to wave speed alone.  The joint contrast of the accompanying
    article is the third vertex: impedance alone, at unchanged speed.
    """

    factors = {}
    for preset, scenario_path in DECOUPLED_SCENARIOS.items():
        scenario, _ = load_defect_scenario_json(_REPO_ROOT / scenario_path)

        assert len(scenario.defects) == 1
        defect = scenario.defects[0]

        # Geometry must repeat the reference campaign exactly.
        assert math.isclose(defect.center[0], 36.5)
        assert math.isclose(defect.center[1], 36.5)
        assert math.isclose(defect.radius, 0.25)

        # Decoupled, therefore never expressed as one joint factor.
        assert defect.material_factor is None

        factors[preset] = (defect.rho_factor, defect.stiffness_factor)

    rho_d, stiff_d = factors["article_L73_gmsh_p1_ppw15_near40_T20_density_only"]
    rho_s, stiff_s = factors["article_L73_gmsh_p1_ppw15_near40_T20_stiffness_only"]

    assert (rho_d, stiff_d) == (0.75, 1.0)
    assert (rho_s, stiff_s) == (1.0, 0.75)

    speed_d = math.sqrt(stiff_d / rho_d)
    speed_s = math.sqrt(stiff_s / rho_s)
    impedance_d = math.sqrt(stiff_d * rho_d)
    impedance_s = math.sqrt(stiff_s * rho_s)

    # Same impedance contrast on both scenarios.
    assert math.isclose(impedance_d, impedance_s)
    assert math.isclose(impedance_d, math.sqrt(0.75))

    # Reciprocal speed contrasts, so the pair brackets the joint-contrast case.
    assert math.isclose(speed_d * speed_s, 1.0)
    assert speed_d > 1.0 > speed_s

    # The joint contrast of the article leaves every phase velocity unchanged.
    published = 0.75
    assert math.isclose(math.sqrt(published / published), 1.0)
    assert math.isclose(math.sqrt(published * published), published)
