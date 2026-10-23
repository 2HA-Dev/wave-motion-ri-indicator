from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from elastic_fwi.postprocessing.residual_wave_overlap import (
    DefectGeometry,
    analyze_overlap_snapshot,
    build_defect_ring_masks,
    compute_nodal_lumped_areas,
    compute_overlap_maps,
)


def structured_triangular_mesh(
    nx: int = 9,
    ny: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)

    xx, yy = np.meshgrid(
        x,
        y,
        indexing="xy",
    )

    points = np.column_stack(
        (
            xx.ravel(),
            yy.ravel(),
        )
    )

    triangles: list[tuple[int, int, int]] = []

    for j in range(ny - 1):
        for i in range(nx - 1):
            lower_left = j * nx + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx
            upper_right = upper_left + 1

            triangles.append(
                (
                    lower_left,
                    lower_right,
                    upper_right,
                )
            )

            triangles.append(
                (
                    lower_left,
                    upper_right,
                    upper_left,
                )
            )

    return (
        points.astype(np.float64),
        np.asarray(
            triangles,
            dtype=np.int64,
        ),
    )


def default_geometry() -> DefectGeometry:
    return DefectGeometry(
        center_x=0.5,
        center_y=0.5,
        radius=0.20,
        ring_width=0.20,
    )


def vector_field(
    amplitude: np.ndarray,
) -> np.ndarray:
    values = np.zeros(
        (amplitude.size, 3),
        dtype=np.float64,
    )

    values[:, 0] = amplitude
    return values


def assert_all_float_metrics_finite(
    result,
) -> None:
    for key, value in asdict(
        result.metrics
    ).items():
        if isinstance(value, float):
            assert np.isfinite(value), key


@pytest.mark.parametrize(
    "zero_reflection,zero_transmission",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_zero_field_gives_zero_interaction(
    zero_reflection: bool,
    zero_transmission: bool,
):
    points, triangles = (
        structured_triangular_mesh()
    )

    amplitude = np.exp(
        -40.0
        * (
            (points[:, 0] - 0.5) ** 2
            + (points[:, 1] - 0.5) ** 2
        )
    )

    reflection = vector_field(amplitude)
    transmission = vector_field(amplitude)

    if zero_reflection:
        reflection[:] = 0.0

    if zero_transmission:
        transmission[:] = 0.0

    result = analyze_overlap_snapshot(
        points=points,
        triangles=triangles,
        reflection_field=reflection,
        transmission_field=transmission,
        geometry=default_geometry(),
        metric_mode="absolute_dot",
    )

    assert np.array_equal(
        result.maps.signed_dot,
        np.zeros(points.shape[0]),
    )

    assert np.array_equal(
        result.maps.absolute_dot,
        np.zeros(points.shape[0]),
    )

    assert np.array_equal(
        result.maps.norm_product,
        np.zeros(points.shape[0]),
    )

    assert result.metrics.global_max == 0.0
    assert result.metrics.energy_total == 0.0
    assert result.metrics.energy_defect == 0.0
    assert result.metrics.energy_ring == 0.0
    assert result.metrics.energy_outside == 0.0
    assert result.metrics.peak_to_artifact_ratio == 0.0
    assert result.metrics.peak_defined is False

    assert_all_float_metrics_finite(result)


def test_overlap_maximum_is_inside_defect():
    points, triangles = (
        structured_triangular_mesh()
    )

    geometry = default_geometry()

    defect, _, _ = build_defect_ring_masks(
        points,
        geometry,
    )

    distance_squared = (
        (points[:, 0] - geometry.center_x) ** 2
        + (points[:, 1] - geometry.center_y) ** 2
    )

    amplitude = np.zeros(
        points.shape[0],
        dtype=np.float64,
    )

    amplitude[defect] = np.exp(
        -80.0 * distance_squared[defect]
    )

    result = analyze_overlap_snapshot(
        points=points,
        triangles=triangles,
        reflection_field=vector_field(amplitude),
        transmission_field=vector_field(amplitude),
        geometry=geometry,
        metric_mode="absolute_dot",
    )

    assert result.metrics.peak_defined is True
    assert (
        result.metrics.global_max_inside_defect
        is True
    )

    assert (
        result.metrics.distance_peak_to_defect_center
        == pytest.approx(0.0)
    )

    assert (
        result.metrics.energy_fraction_defect
        == pytest.approx(1.0)
    )

    assert result.metrics.energy_outside == 0.0
    assert result.metrics.peak_outside == 0.0
    assert result.metrics.peak_to_artifact_ratio > 1.0


def test_overlap_concentrated_outside_defect():
    points, triangles = (
        structured_triangular_mesh()
    )

    amplitude = np.zeros(
        points.shape[0],
        dtype=np.float64,
    )

    corner = int(
        np.argmin(
            points[:, 0] + points[:, 1]
        )
    )

    amplitude[corner] = 1.0

    result = analyze_overlap_snapshot(
        points=points,
        triangles=triangles,
        reflection_field=vector_field(amplitude),
        transmission_field=vector_field(amplitude),
        geometry=default_geometry(),
        metric_mode="norm_product",
    )

    assert result.metrics.peak_defined is True
    assert (
        result.metrics.global_max_inside_defect
        is False
    )

    assert (
        result.metrics.global_max_inside_ring
        is False
    )

    assert (
        result.metrics.energy_fraction_defect
        == pytest.approx(0.0)
    )

    assert result.metrics.peak_defect == 0.0
    assert result.metrics.peak_outside > 0.0
    assert (
        result.metrics.peak_to_artifact_ratio
        == pytest.approx(0.0)
    )


def test_positive_scaling_preserves_normalized_maps():
    points, triangles = (
        structured_triangular_mesh()
    )

    geometry = default_geometry()

    reflection_amplitude = np.exp(
        -30.0
        * (
            (points[:, 0] - 0.47) ** 2
            + (points[:, 1] - 0.52) ** 2
        )
    )

    transmission_amplitude = np.exp(
        -35.0
        * (
            (points[:, 0] - 0.53) ** 2
            + (points[:, 1] - 0.48) ** 2
        )
    )

    reflection = vector_field(
        reflection_amplitude
    )

    transmission = vector_field(
        transmission_amplitude
    )

    scale = 7.5

    maps_reference = compute_overlap_maps(
        reflection,
        transmission,
    )

    maps_scaled = compute_overlap_maps(
        scale * reflection,
        transmission,
    )

    assert np.allclose(
        maps_scaled.signed_dot,
        scale * maps_reference.signed_dot,
    )

    assert np.allclose(
        maps_scaled.absolute_dot,
        scale * maps_reference.absolute_dot,
    )

    assert np.allclose(
        maps_scaled.norm_product,
        scale * maps_reference.norm_product,
    )

    assert np.allclose(
        maps_scaled.normalized_signed_dot,
        maps_reference.normalized_signed_dot,
    )

    assert np.allclose(
        maps_scaled.normalized_absolute_dot,
        maps_reference.normalized_absolute_dot,
    )

    assert np.allclose(
        maps_scaled.normalized_norm_product,
        maps_reference.normalized_norm_product,
    )

    result_reference = analyze_overlap_snapshot(
        points=points,
        triangles=triangles,
        reflection_field=reflection,
        transmission_field=transmission,
        geometry=geometry,
        metric_mode="absolute_dot",
    )

    result_scaled = analyze_overlap_snapshot(
        points=points,
        triangles=triangles,
        reflection_field=scale * reflection,
        transmission_field=transmission,
        geometry=geometry,
        metric_mode="absolute_dot",
    )

    assert (
        result_scaled.metrics.energy_total
        == pytest.approx(
            scale
            * result_reference.metrics.energy_total
        )
    )

    assert (
        result_scaled.metrics.global_max
        == pytest.approx(
            scale
            * result_reference.metrics.global_max
        )
    )

    assert (
        result_scaled.metrics.energy_fraction_defect
        == pytest.approx(
            result_reference.metrics.energy_fraction_defect
        )
    )


def test_lumped_weights_integrate_unit_square():
    points, triangles = (
        structured_triangular_mesh(
            nx=7,
            ny=11,
        )
    )

    weights = compute_nodal_lumped_areas(
        points,
        triangles,
    )

    assert np.all(weights >= 0.0)

    assert float(np.sum(weights)) == pytest.approx(
        1.0,
        rel=1.0e-14,
        abs=1.0e-14,
    )


def test_reproducibility_is_exact():
    points, triangles = (
        structured_triangular_mesh()
    )

    amplitude_1 = np.sin(
        np.pi * points[:, 0]
    ) * np.sin(
        np.pi * points[:, 1]
    )

    amplitude_2 = np.cos(
        0.5 * np.pi * points[:, 0]
    ) * np.sin(
        np.pi * points[:, 1]
    )

    arguments = {
        "points": points,
        "triangles": triangles,
        "reflection_field": vector_field(
            amplitude_1
        ),
        "transmission_field": vector_field(
            amplitude_2
        ),
        "geometry": default_geometry(),
        "metric_mode": "absolute_dot",
    }

    first = analyze_overlap_snapshot(
        **arguments
    )

    second = analyze_overlap_snapshot(
        **arguments
    )

    assert np.array_equal(
        first.maps.signed_dot,
        second.maps.signed_dot,
    )

    assert np.array_equal(
        first.maps.absolute_dot,
        second.maps.absolute_dot,
    )

    assert np.array_equal(
        first.maps.norm_product,
        second.maps.norm_product,
    )

    assert first.metrics.as_dict() == second.metrics.as_dict()


def test_incompatible_field_shapes_are_rejected():
    points, triangles = (
        structured_triangular_mesh()
    )

    reflection = np.zeros(
        (points.shape[0], 3)
    )

    transmission = np.zeros(
        (points.shape[0] - 1, 3)
    )

    with pytest.raises(
        ValueError,
        match="incompatible",
    ):
        analyze_overlap_snapshot(
            points=points,
            triangles=triangles,
            reflection_field=reflection,
            transmission_field=transmission,
            geometry=default_geometry(),
        )


def test_non_finite_field_is_rejected():
    points, triangles = (
        structured_triangular_mesh()
    )

    reflection = np.zeros(
        (points.shape[0], 3)
    )

    transmission = np.zeros_like(
        reflection
    )

    reflection[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        analyze_overlap_snapshot(
            points=points,
            triangles=triangles,
            reflection_field=reflection,
            transmission_field=transmission,
            geometry=default_geometry(),
        )


def test_empty_defect_mask_is_rejected():
    points, triangles = (
        structured_triangular_mesh()
    )

    field = np.zeros(
        (points.shape[0], 3)
    )

    geometry = DefectGeometry(
        center_x=10.0,
        center_y=10.0,
        radius=0.01,
        ring_width=0.1,
    )

    with pytest.raises(
        ValueError,
        match="defect mask is empty",
    ):
        analyze_overlap_snapshot(
            points=points,
            triangles=triangles,
            reflection_field=field,
            transmission_field=field,
            geometry=geometry,
        )


@pytest.mark.parametrize(
    "ring_width",
    [0.0, -0.1],
)
def test_invalid_ring_width_is_rejected(
    ring_width: float,
):
    points, triangles = (
        structured_triangular_mesh()
    )

    field = np.zeros(
        (points.shape[0], 3)
    )

    geometry = DefectGeometry(
        center_x=0.5,
        center_y=0.5,
        radius=0.2,
        ring_width=ring_width,
    )

    with pytest.raises(
        ValueError,
        match="Ring width",
    ):
        analyze_overlap_snapshot(
            points=points,
            triangles=triangles,
            reflection_field=field,
            transmission_field=field,
            geometry=geometry,
        )


def test_significant_third_component_is_rejected():
    points, triangles = (
        structured_triangular_mesh()
    )

    reflection = np.zeros(
        (points.shape[0], 3),
        dtype=np.float64,
    )

    transmission = np.zeros_like(
        reflection
    )

    reflection[:, 0] = 1.0
    transmission[:, 0] = 1.0

    reflection[:, 2] = 1.0e-3

    with pytest.raises(
        ValueError,
        match="third component",
    ):
        analyze_overlap_snapshot(
            points=points,
            triangles=triangles,
            reflection_field=reflection,
            transmission_field=transmission,
            geometry=default_geometry(),
        )


def test_signed_mode_is_not_accepted_for_energy_metrics():
    points, triangles = (
        structured_triangular_mesh()
    )

    field = np.zeros(
        (points.shape[0], 3)
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        analyze_overlap_snapshot(
            points=points,
            triangles=triangles,
            reflection_field=field,
            transmission_field=field,
            geometry=default_geometry(),
            metric_mode="signed_dot",
        )
