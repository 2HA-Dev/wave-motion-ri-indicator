from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


InteractionMode = Literal[
    "signed_dot",
    "absolute_dot",
    "norm_product",
]

MetricMode = Literal[
    "absolute_dot",
    "norm_product",
]


_INTERACTION_ALIASES: dict[str, InteractionMode] = {
    "signed_dot": "signed_dot",
    "signed-dot": "signed_dot",
    "absolute_dot": "absolute_dot",
    "absolute-dot": "absolute_dot",
    "abs_dot": "absolute_dot",
    "abs-dot": "absolute_dot",
    "norm_product": "norm_product",
    "norm-product": "norm_product",
    "energy": "norm_product",
}


_INTERACTION_DEFINITIONS: dict[str, str] = {
    "signed_dot": (
        "W1_xy dot Lambda2_xy"
    ),
    "absolute_dot": (
        "abs(W1_xy dot Lambda2_xy)"
    ),
    "norm_product": (
        "norm(W1_xy) * norm(Lambda2_xy)"
    ),
}


@dataclass(frozen=True)
class DefectGeometry:
    center_x: float
    center_y: float
    radius: float
    ring_width: float

    def validate(self) -> None:
        values = np.asarray(
            [
                self.center_x,
                self.center_y,
                self.radius,
                self.ring_width,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Defect geometry contains non-finite values."
            )

        if self.radius <= 0.0:
            raise ValueError(
                "Defect radius must be strictly positive."
            )

        if self.ring_width <= 0.0:
            raise ValueError(
                "Ring width must be strictly positive."
            )


@dataclass(frozen=True)
class OverlapMaps:
    signed_dot: np.ndarray
    absolute_dot: np.ndarray
    norm_product: np.ndarray
    normalized_signed_dot: np.ndarray
    normalized_absolute_dot: np.ndarray
    normalized_norm_product: np.ndarray

    def get(
        self,
        mode: InteractionMode | str,
    ) -> np.ndarray:
        canonical = canonical_interaction_mode(mode)

        if canonical == "signed_dot":
            return self.signed_dot

        if canonical == "absolute_dot":
            return self.absolute_dot

        return self.norm_product

    def get_normalized(
        self,
        mode: InteractionMode | str,
    ) -> np.ndarray:
        canonical = canonical_interaction_mode(mode)

        if canonical == "signed_dot":
            return self.normalized_signed_dot

        if canonical == "absolute_dot":
            return self.normalized_absolute_dot

        return self.normalized_norm_product


@dataclass(frozen=True)
class OverlapMetrics:
    interaction_definition: str
    metric_mode: str
    normalization_mode: str
    peak_defined: bool

    global_max: float
    global_max_x: float
    global_max_y: float
    global_max_inside_defect: bool
    global_max_inside_ring: bool
    distance_peak_to_defect_center: float

    weighted_centroid_x: float
    weighted_centroid_y: float
    distance_centroid_to_defect_center: float

    area_total: float
    area_defect: float
    area_ring: float
    area_outside: float

    energy_total: float
    energy_defect: float
    energy_ring: float
    energy_outside: float

    energy_fraction_defect: float
    energy_fraction_defect_and_ring: float
    energy_inside_outside_ratio: float

    peak_defect: float
    peak_ring: float
    peak_outside: float
    peak_to_artifact_ratio: float
    peak_outside_region: str

    ratio_epsilon: float
    energy_denominator_floor: float
    peak_denominator_floor: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OverlapSnapshotResult:
    maps: OverlapMaps
    nodal_weights: np.ndarray
    defect_mask: np.ndarray
    ring_mask: np.ndarray
    outside_mask: np.ndarray
    metrics: OverlapMetrics


def canonical_interaction_mode(
    mode: InteractionMode | str,
) -> InteractionMode:
    normalized = str(mode).strip().lower()

    try:
        return _INTERACTION_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            "Interaction mode must be one of "
            "'signed_dot', 'absolute_dot', or "
            "'norm_product'."
        ) from exc


def validate_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        points,
        dtype=np.float64,
    )

    connectivity_raw = np.asarray(triangles)

    if coordinates.ndim != 2:
        raise ValueError(
            "Points must be a two-dimensional array."
        )

    if coordinates.shape[1] < 2:
        raise ValueError(
            "Points must expose at least x and y coordinates."
        )

    if coordinates.shape[0] == 0:
        raise ValueError("Mesh contains no points.")

    if not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "Mesh coordinates contain non-finite values."
        )

    if connectivity_raw.ndim != 2:
        raise ValueError(
            "Triangle connectivity must be two-dimensional."
        )

    if connectivity_raw.shape[1] != 3:
        raise ValueError(
            "Only three-node P1 triangles are supported."
        )

    if connectivity_raw.shape[0] == 0:
        raise ValueError("Mesh contains no triangles.")

    if not np.issubdtype(
        connectivity_raw.dtype,
        np.integer,
    ):
        raise ValueError(
            "Triangle connectivity must use integer indices."
        )

    connectivity = np.asarray(
        connectivity_raw,
        dtype=np.int64,
    )

    if np.min(connectivity) < 0:
        raise ValueError(
            "Triangle connectivity contains negative indices."
        )

    if np.max(connectivity) >= coordinates.shape[0]:
        raise ValueError(
            "Triangle connectivity references an unknown point."
        )

    xy = np.ascontiguousarray(
        coordinates[:, :2],
        dtype=np.float64,
    )

    return xy, connectivity


def extract_physical_xy_components(
    reflection_field: np.ndarray,
    transmission_field: np.ndarray,
    *,
    z_atol: float = 0.0,
    z_rtol: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    reflection = np.asarray(
        reflection_field,
        dtype=np.float64,
    )

    transmission = np.asarray(
        transmission_field,
        dtype=np.float64,
    )

    if reflection.ndim != 2:
        raise ValueError(
            "Reflection field must be a two-dimensional array."
        )

    if transmission.ndim != 2:
        raise ValueError(
            "Transmission field must be a two-dimensional array."
        )

    if reflection.shape != transmission.shape:
        raise ValueError(
            "Reflection and transmission fields have "
            "incompatible dimensions."
        )

    if reflection.shape[0] == 0:
        raise ValueError("Fields contain no nodal values.")

    if reflection.shape[1] not in (2, 3):
        raise ValueError(
            "Fields must expose two physical components or "
            "three VTK components."
        )

    if not np.all(np.isfinite(reflection)):
        raise ValueError(
            "Reflection field contains non-finite values."
        )

    if not np.all(np.isfinite(transmission)):
        raise ValueError(
            "Transmission field contains non-finite values."
        )

    if not np.isfinite(z_atol) or z_atol < 0.0:
        raise ValueError(
            "z_atol must be finite and non-negative."
        )

    if not np.isfinite(z_rtol) or z_rtol < 0.0:
        raise ValueError(
            "z_rtol must be finite and non-negative."
        )

    if reflection.shape[1] == 3:
        for name, field in (
            ("reflection", reflection),
            ("transmission", transmission),
        ):
            xy_norm = np.sqrt(
                np.sum(
                    field[:, :2] * field[:, :2],
                    axis=1,
                )
            )

            xy_scale = float(
                np.max(xy_norm)
            )

            z_scale = float(
                np.max(np.abs(field[:, 2]))
            )

            tolerance = (
                float(z_atol)
                + float(z_rtol) * xy_scale
            )

            if z_scale > tolerance:
                raise ValueError(
                    f"The third component of the {name} field "
                    f"is significant: max_abs_z={z_scale:.16e}, "
                    f"tolerance={tolerance:.16e}."
                )

    return (
        np.ascontiguousarray(
            reflection[:, :2],
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            transmission[:, :2],
            dtype=np.float64,
        ),
    )


def compute_triangle_areas(
    points: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    xy, connectivity = validate_mesh(
        points,
        triangles,
    )

    vertices = xy[connectivity]

    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]

    signed_twice_area = (
        edge_1[:, 0] * edge_2[:, 1]
        - edge_1[:, 1] * edge_2[:, 0]
    )

    areas = 0.5 * np.abs(
        signed_twice_area
    )

    if not np.all(np.isfinite(areas)):
        raise ValueError(
            "Computed triangle areas are not finite."
        )

    if np.any(areas <= 0.0):
        raise ValueError(
            "Mesh contains a degenerate triangle."
        )

    return areas


def compute_nodal_lumped_areas(
    points: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    xy, connectivity = validate_mesh(
        points,
        triangles,
    )

    areas = compute_triangle_areas(
        xy,
        connectivity,
    )

    weights = np.zeros(
        xy.shape[0],
        dtype=np.float64,
    )

    contribution = areas / 3.0

    for local_vertex in range(3):
        np.add.at(
            weights,
            connectivity[:, local_vertex],
            contribution,
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "Computed nodal weights are not finite."
        )

    if np.any(weights < 0.0):
        raise ValueError(
            "Computed nodal weights contain negative values."
        )

    if float(np.sum(weights)) <= 0.0:
        raise ValueError(
            "Total mesh area is not strictly positive."
        )

    return weights


def build_defect_ring_masks(
    points: np.ndarray,
    geometry: DefectGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry.validate()

    coordinates = np.asarray(
        points,
        dtype=np.float64,
    )

    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(
            "Points must expose x and y coordinates."
        )

    if not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "Mesh coordinates contain non-finite values."
        )

    delta_x = (
        coordinates[:, 0]
        - float(geometry.center_x)
    )

    delta_y = (
        coordinates[:, 1]
        - float(geometry.center_y)
    )

    distances = np.sqrt(
        delta_x * delta_x
        + delta_y * delta_y
    )

    scale = max(
        1.0,
        abs(float(geometry.center_x)),
        abs(float(geometry.center_y)),
        float(geometry.radius),
        float(geometry.ring_width),
        float(np.max(np.abs(coordinates[:, :2]))),
    )

    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * scale
    )

    defect = (
        distances
        <= float(geometry.radius) + tolerance
    )

    ring_outer_radius = (
        float(geometry.radius)
        + float(geometry.ring_width)
    )

    ring = (
        (~defect)
        & (
            distances
            <= ring_outer_radius + tolerance
        )
    )

    outside = ~(defect | ring)

    if not np.any(defect):
        raise ValueError(
            "The defect mask is empty on the supplied mesh."
        )

    return defect, ring, outside


def normalize_interaction_map(
    values: np.ndarray,
    *,
    signed: bool,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            "Interaction map must be one-dimensional."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Interaction map contains non-finite values."
        )

    if signed:
        scale = float(
            np.max(np.abs(array))
        )
    else:
        scale = float(
            np.max(array)
        )

    if scale <= 0.0:
        return np.zeros_like(array)

    return array / scale


def compute_overlap_maps(
    reflection_field: np.ndarray,
    transmission_field: np.ndarray,
    *,
    z_atol: float = 0.0,
    z_rtol: float = 1.0e-14,
) -> OverlapMaps:
    reflection_xy, transmission_xy = (
        extract_physical_xy_components(
            reflection_field,
            transmission_field,
            z_atol=z_atol,
            z_rtol=z_rtol,
        )
    )

    signed_dot = np.einsum(
        "ij,ij->i",
        reflection_xy,
        transmission_xy,
        optimize=True,
    )

    absolute_dot = np.abs(signed_dot)

    reflection_norm = np.sqrt(
        np.einsum(
            "ij,ij->i",
            reflection_xy,
            reflection_xy,
            optimize=True,
        )
    )

    transmission_norm = np.sqrt(
        np.einsum(
            "ij,ij->i",
            transmission_xy,
            transmission_xy,
            optimize=True,
        )
    )

    norm_product = (
        reflection_norm
        * transmission_norm
    )

    return OverlapMaps(
        signed_dot=signed_dot,
        absolute_dot=absolute_dot,
        norm_product=norm_product,
        normalized_signed_dot=(
            normalize_interaction_map(
                signed_dot,
                signed=True,
            )
        ),
        normalized_absolute_dot=(
            normalize_interaction_map(
                absolute_dot,
                signed=False,
            )
        ),
        normalized_norm_product=(
            normalize_interaction_map(
                norm_product,
                signed=False,
            )
        ),
    )


def _masked_max(
    values: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not np.any(mask):
        return 0.0

    return float(
        np.max(values[mask])
    )


def analyze_overlap_snapshot(
    *,
    points: np.ndarray,
    triangles: np.ndarray,
    reflection_field: np.ndarray,
    transmission_field: np.ndarray,
    geometry: DefectGeometry,
    metric_mode: MetricMode | str = "absolute_dot",
    ratio_epsilon: float = 1.0e-15,
    z_atol: float = 0.0,
    z_rtol: float = 1.0e-14,
) -> OverlapSnapshotResult:
    xy, connectivity = validate_mesh(
        points,
        triangles,
    )

    if not np.isfinite(ratio_epsilon):
        raise ValueError(
            "ratio_epsilon must be finite."
        )

    if ratio_epsilon <= 0.0:
        raise ValueError(
            "ratio_epsilon must be strictly positive."
        )

    canonical_mode = canonical_interaction_mode(
        metric_mode
    )

    if canonical_mode == "signed_dot":
        raise ValueError(
            "Physical concentration metrics require a "
            "non-negative quantity. Use 'absolute_dot' or "
            "'norm_product'."
        )

    if reflection_field.shape[0] != xy.shape[0]:
        raise ValueError(
            "Reflection field size is incompatible with "
            "the mesh."
        )

    if transmission_field.shape[0] != xy.shape[0]:
        raise ValueError(
            "Transmission field size is incompatible with "
            "the mesh."
        )

    maps = compute_overlap_maps(
        reflection_field,
        transmission_field,
        z_atol=z_atol,
        z_rtol=z_rtol,
    )

    measure = maps.get(canonical_mode)

    if np.any(measure < 0.0):
        raise RuntimeError(
            "Metric interaction quantity must be non-negative."
        )

    nodal_weights = compute_nodal_lumped_areas(
        xy,
        connectivity,
    )

    defect_mask, ring_mask, outside_mask = (
        build_defect_ring_masks(
            xy,
            geometry,
        )
    )

    weighted_measure = (
        nodal_weights
        * measure
    )

    energy_total = float(
        np.sum(weighted_measure)
    )

    energy_defect = float(
        np.sum(
            weighted_measure[defect_mask]
        )
    )

    energy_ring = float(
        np.sum(
            weighted_measure[ring_mask]
        )
    )

    energy_outside = float(
        np.sum(
            weighted_measure[outside_mask]
        )
    )

    area_total = float(
        np.sum(nodal_weights)
    )

    area_defect = float(
        np.sum(nodal_weights[defect_mask])
    )

    area_ring = float(
        np.sum(nodal_weights[ring_mask])
    )

    area_outside = float(
        np.sum(nodal_weights[outside_mask])
    )

    global_max = float(
        np.max(measure)
    )

    if global_max > 0.0:
        peak_defined = True

        peak_index = int(
            np.argmax(measure)
        )

        peak_x = float(
            xy[peak_index, 0]
        )

        peak_y = float(
            xy[peak_index, 1]
        )

        peak_inside_defect = bool(
            defect_mask[peak_index]
        )

        peak_inside_ring = bool(
            ring_mask[peak_index]
        )
    else:
        peak_defined = False
        peak_x = float(geometry.center_x)
        peak_y = float(geometry.center_y)
        peak_inside_defect = True
        peak_inside_ring = False

    distance_peak = float(
        np.hypot(
            peak_x - float(geometry.center_x),
            peak_y - float(geometry.center_y),
        )
    )

    if energy_total > 0.0:
        centroid_x = float(
            np.sum(
                xy[:, 0] * weighted_measure
            )
            / energy_total
        )

        centroid_y = float(
            np.sum(
                xy[:, 1] * weighted_measure
            )
            / energy_total
        )
    else:
        centroid_x = float(geometry.center_x)
        centroid_y = float(geometry.center_y)

    distance_centroid = float(
        np.hypot(
            centroid_x - float(geometry.center_x),
            centroid_y - float(geometry.center_y),
        )
    )

    if energy_total > 0.0:
        energy_fraction_defect = (
            energy_defect / energy_total
        )

        energy_fraction_defect_and_ring = (
            (energy_defect + energy_ring)
            / energy_total
        )

        energy_denominator_floor = (
            float(ratio_epsilon)
            * energy_total
        )
    else:
        energy_fraction_defect = 0.0
        energy_fraction_defect_and_ring = 0.0
        energy_denominator_floor = 1.0

    energy_denominator = max(
        energy_outside,
        energy_denominator_floor,
    )

    if energy_total > 0.0:
        energy_inside_outside_ratio = (
            energy_defect
            / energy_denominator
        )
    else:
        energy_inside_outside_ratio = 0.0

    peak_defect = _masked_max(
        measure,
        defect_mask,
    )

    peak_ring = _masked_max(
        measure,
        ring_mask,
    )

    peak_outside = _masked_max(
        measure,
        outside_mask,
    )

    if global_max > 0.0:
        peak_denominator_floor = (
            float(ratio_epsilon)
            * global_max
        )
    else:
        peak_denominator_floor = 1.0

    peak_denominator = max(
        peak_outside,
        peak_denominator_floor,
    )

    if global_max > 0.0:
        peak_to_artifact_ratio = (
            peak_defect
            / peak_denominator
        )
    else:
        peak_to_artifact_ratio = 0.0

    metrics = OverlapMetrics(
        interaction_definition=(
            _INTERACTION_DEFINITIONS[
                canonical_mode
            ]
        ),
        metric_mode=canonical_mode,
        normalization_mode="maximum",
        peak_defined=peak_defined,
        global_max=global_max,
        global_max_x=peak_x,
        global_max_y=peak_y,
        global_max_inside_defect=(
            peak_inside_defect
        ),
        global_max_inside_ring=(
            peak_inside_ring
        ),
        distance_peak_to_defect_center=(
            distance_peak
        ),
        weighted_centroid_x=centroid_x,
        weighted_centroid_y=centroid_y,
        distance_centroid_to_defect_center=(
            distance_centroid
        ),
        area_total=area_total,
        area_defect=area_defect,
        area_ring=area_ring,
        area_outside=area_outside,
        energy_total=energy_total,
        energy_defect=energy_defect,
        energy_ring=energy_ring,
        energy_outside=energy_outside,
        energy_fraction_defect=(
            energy_fraction_defect
        ),
        energy_fraction_defect_and_ring=(
            energy_fraction_defect_and_ring
        ),
        energy_inside_outside_ratio=(
            energy_inside_outside_ratio
        ),
        peak_defect=peak_defect,
        peak_ring=peak_ring,
        peak_outside=peak_outside,
        peak_to_artifact_ratio=(
            peak_to_artifact_ratio
        ),
        peak_outside_region=(
            "outside_defect_and_ring"
        ),
        ratio_epsilon=float(ratio_epsilon),
        energy_denominator_floor=(
            energy_denominator_floor
        ),
        peak_denominator_floor=(
            peak_denominator_floor
        ),
    )

    metric_values = metrics.as_dict()

    for key, value in metric_values.items():
        if isinstance(value, float):
            if not np.isfinite(value):
                raise RuntimeError(
                    f"Metric {key!r} is not finite."
                )

    return OverlapSnapshotResult(
        maps=maps,
        nodal_weights=nodal_weights,
        defect_mask=defect_mask,
        ring_mask=ring_mask,
        outside_mask=outside_mask,
        metrics=metrics,
    )
