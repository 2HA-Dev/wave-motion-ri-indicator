"""Numerical diagnostics for the mirror-symmetry theorem of the C66 kernels.

The theorem, established symbolically in
``scripts/revision_wave_motion/verify_mirror_symmetry_symbolic.py``, predicts three
measurable properties of the signed DG0 sensitivity densities of the ``line``
and ``box`` configurations, whose geometry is invariant under the mirror

    M : x1 -> 73 - x1,   fixed axis x1 = 36.5.

P1  *Every* signed kernel is mirror even, because each is a product of two
    strain factors of equal parity.  This is a global check on the whole
    pipeline; its accuracy is limited by the mesh, which gmsh does not generate
    symmetrically, so the measured odd fraction is a discretisation floor and
    not a physical quantity.

P2  The four ``C66`` kernels (classical, reflection-only, transmission-only
    and residual-interaction) vanish identically on the axis, whereas the
    kernels of ``rho``, ``C11``, ``C22`` and ``C12`` do not.  This is the
    property specific to the shear coefficient.

P3  Near the axis the ``C66`` kernels vanish *quadratically* in the distance to
    it, because each shear factor vanishes linearly.  The normal-strain kernels
    are flat there (exponent 0).

This module computes the three corresponding numbers from cell-centred DG0
fields.  It performs no PDE solve and reads only already-exported maps.

All spatial integrals follow the DG0/L2 density convention: values are
cell-average densities and every statistic is weighted by the cell measure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


# Geometry of the article-scale campaigns: square domain [0, 73]^2 with the
# defect and both arrays centred on x1 = 36.5.
DEFAULT_AXIS_POSITION = 36.5
DEFAULT_DEFECT_CENTER = (36.5, 36.5)
DEFAULT_DEFECT_RADIUS = 0.25

# Above this floor-to-peak ratio the power-law component of the decay model is
# not resolved and its exponent carries no information.
FLOOR_DOMINANCE_LIMIT = 0.5


class MirrorDiagnosticsError(ValueError):
    """Raised when the requested diagnostic cannot be computed."""


@dataclass(frozen=True)
class MirrorParityMetrics:
    """Even/odd split of a field under the mirror."""

    num_cells: int
    num_paired_cells: int
    pairing_distance_median: float
    pairing_distance_p95: float
    odd_l2_fraction: float
    even_l2_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AxisVanishingMetrics:
    """Amplitude on the mirror axis, relative to a nearby reference band."""

    axis_band_half_width: float
    reference_band_min: float
    reference_band_max: float
    span_half_height: float
    num_axis_cells: int
    num_reference_cells: int
    axis_rms: float
    reference_rms: float
    axis_suppression_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransverseDecayMetrics:
    """Log-log slope of the field amplitude against distance to the axis.

    Two exponents are reported.  ``decay_exponent`` is the plain log-log slope
    over the fit window.  ``floor_corrected_exponent`` comes from the
    three-parameter model

        rms(s)^2 = amplitude^2 * s^(2 p) + floor^2,

    which separates the physical power law from the non-vanishing numerical
    residual left on the axis by a mesh that gmsh does not generate
    symmetrically.  The plain slope is biased low whenever the floor is not
    negligible over the window; the corrected exponent is the one to compare
    against the predicted value of 2.
    """

    fit_min_distance: float
    fit_max_distance: float
    num_bins_used: int
    decay_exponent: float
    decay_exponent_stderr: float
    fit_r_squared: float
    floor_corrected_exponent: float
    floor_corrected_exponent_stderr: float
    floor_amplitude: float
    floor_to_peak_ratio: float
    floor_model_converged: bool
    bin_centers: tuple[float, ...]
    bin_rms: tuple[float, ...]
    bin_cell_counts: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_inputs(
    centers: np.ndarray,
    values: np.ndarray,
    measures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    measures = np.asarray(measures, dtype=np.float64)

    if centers.ndim != 2 or centers.shape[1] < 2:
        raise MirrorDiagnosticsError("centers must have shape (n, >=2).")
    if values.shape[0] != centers.shape[0]:
        raise MirrorDiagnosticsError("values and centers must have the same length.")
    if measures.shape[0] != centers.shape[0]:
        raise MirrorDiagnosticsError("measures and centers must have the same length.")
    if not np.all(np.isfinite(measures)) or np.any(measures <= 0.0):
        raise MirrorDiagnosticsError("cell measures must be finite and positive.")

    return centers[:, :2], values, measures


def _weighted_rms(values: np.ndarray, measures: np.ndarray) -> float:
    total = float(np.sum(measures, dtype=np.float64))
    if total <= 0.0:
        return math.nan
    energy = float(np.sum(measures * values * values, dtype=np.float64))
    return math.sqrt(max(energy, 0.0) / total)


def mirror_parity_metrics(
    centers: np.ndarray,
    values: np.ndarray,
    measures: np.ndarray,
    *,
    axis_position: float = DEFAULT_AXIS_POSITION,
    max_pairing_distance: float | None = None,
) -> MirrorParityMetrics:
    """Split a cell field into its mirror-even and mirror-odd parts (P1).

    Each cell is paired with the cell whose centre is nearest to its mirror
    image.  Because the mesh is not generated symmetrically, the pairing is
    approximate; ``pairing_distance_median`` and ``pairing_distance_p95`` report
    how approximate, and bound the accuracy of ``odd_l2_fraction``.
    """

    centers, values, measures = _validate_inputs(centers, values, measures)

    mirrored = centers.copy()
    mirrored[:, 0] = 2.0 * float(axis_position) - centers[:, 0]

    tree = cKDTree(centers)
    distances, partners = tree.query(mirrored, k=1)

    keep = np.ones(centers.shape[0], dtype=bool)
    if max_pairing_distance is not None:
        keep = distances <= float(max_pairing_distance)
        if not np.any(keep):
            raise MirrorDiagnosticsError(
                "No cell pair survives the requested max_pairing_distance."
            )

    partner_values = values[partners]
    odd_part = 0.5 * (values - partner_values)
    even_part = 0.5 * (values + partner_values)

    total_rms = _weighted_rms(values[keep], measures[keep])
    odd_rms = _weighted_rms(odd_part[keep], measures[keep])
    even_rms = _weighted_rms(even_part[keep], measures[keep])

    if not math.isfinite(total_rms) or total_rms <= 0.0:
        odd_fraction = math.nan
        even_fraction = math.nan
    else:
        odd_fraction = odd_rms / total_rms
        even_fraction = even_rms / total_rms

    return MirrorParityMetrics(
        num_cells=int(centers.shape[0]),
        num_paired_cells=int(np.count_nonzero(keep)),
        pairing_distance_median=float(np.median(distances[keep])),
        pairing_distance_p95=float(np.percentile(distances[keep], 95.0)),
        odd_l2_fraction=float(odd_fraction),
        even_l2_fraction=float(even_fraction),
    )


def axis_vanishing_metrics(
    centers: np.ndarray,
    values: np.ndarray,
    measures: np.ndarray,
    *,
    axis_position: float = DEFAULT_AXIS_POSITION,
    defect_center: tuple[float, float] = DEFAULT_DEFECT_CENTER,
    axis_band_half_width: float = 0.02,
    reference_band_min: float = 0.10,
    reference_band_max: float = 0.30,
    span_half_height: float = 0.75,
) -> AxisVanishingMetrics:
    """Amplitude on the axis relative to a nearby off-axis band (P2).

    The comparison is restricted to ``|x2 - defect_center[1]| <=
    span_half_height`` so that both bands sample the same uniformly refined
    region around the defect and the same range of the propagation coordinate.
    The returned ``axis_suppression_ratio`` is dimensionless and invariant under
    the per-map normalisation, so it is comparable across coefficients and
    modalities.
    """

    centers, values, measures = _validate_inputs(centers, values, measures)

    if axis_band_half_width <= 0.0:
        raise MirrorDiagnosticsError("axis_band_half_width must be positive.")
    if not reference_band_min < reference_band_max:
        raise MirrorDiagnosticsError(
            "reference_band_min must be smaller than reference_band_max."
        )

    distance_to_axis = np.abs(centers[:, 0] - float(axis_position))
    within_span = np.abs(centers[:, 1] - float(defect_center[1])) <= float(
        span_half_height
    )

    axis_mask = within_span & (distance_to_axis <= float(axis_band_half_width))
    reference_mask = (
        within_span
        & (distance_to_axis >= float(reference_band_min))
        & (distance_to_axis <= float(reference_band_max))
    )

    if not np.any(axis_mask):
        raise MirrorDiagnosticsError("The axis band contains no cell.")
    if not np.any(reference_mask):
        raise MirrorDiagnosticsError("The reference band contains no cell.")

    axis_rms = _weighted_rms(values[axis_mask], measures[axis_mask])
    reference_rms = _weighted_rms(values[reference_mask], measures[reference_mask])

    ratio = (
        axis_rms / reference_rms
        if math.isfinite(reference_rms) and reference_rms > 0.0
        else math.nan
    )

    return AxisVanishingMetrics(
        axis_band_half_width=float(axis_band_half_width),
        reference_band_min=float(reference_band_min),
        reference_band_max=float(reference_band_max),
        span_half_height=float(span_half_height),
        num_axis_cells=int(np.count_nonzero(axis_mask)),
        num_reference_cells=int(np.count_nonzero(reference_mask)),
        axis_rms=float(axis_rms),
        reference_rms=float(reference_rms),
        axis_suppression_ratio=float(ratio),
    )


def transverse_decay_metrics(
    centers: np.ndarray,
    values: np.ndarray,
    measures: np.ndarray,
    *,
    axis_position: float = DEFAULT_AXIS_POSITION,
    defect_center: tuple[float, float] = DEFAULT_DEFECT_CENTER,
    span_half_height: float = 0.75,
    num_bins: int = 24,
    max_distance: float = 0.24,
    fit_min_distance: float = 0.03,
    fit_max_distance: float = 0.15,
    min_cells_per_bin: int = 8,
) -> TransverseDecayMetrics:
    """Log-log slope of the amplitude against the distance to the axis (P3).

    The prediction is an exponent close to 2 for the ``C66`` kernels and close
    to 0 for the normal-strain kernels.  The fit window defaults to distances
    between roughly one cell diameter and one sixth of the shortest wavelength,
    which is where the linearisation of each shear factor about the axis holds
    and where the mesh is uniformly refined.
    """

    centers, values, measures = _validate_inputs(centers, values, measures)

    if max_distance <= 0.0 or num_bins < 2:
        raise MirrorDiagnosticsError("max_distance must be positive and num_bins >= 2.")
    if not fit_min_distance < fit_max_distance:
        raise MirrorDiagnosticsError(
            "fit_min_distance must be smaller than fit_max_distance."
        )

    distance_to_axis = np.abs(centers[:, 0] - float(axis_position))
    within_span = np.abs(centers[:, 1] - float(defect_center[1])) <= float(
        span_half_height
    )
    selected = within_span & (distance_to_axis <= float(max_distance))
    if not np.any(selected):
        raise MirrorDiagnosticsError("No cell falls inside the analysis strip.")

    edges = np.linspace(0.0, float(max_distance), int(num_bins) + 1)
    bin_index = np.digitize(distance_to_axis[selected], edges) - 1
    bin_index = np.clip(bin_index, 0, int(num_bins) - 1)

    sel_values = values[selected]
    sel_measures = measures[selected]

    centers_out: list[float] = []
    rms_out: list[float] = []
    counts_out: list[int] = []
    for index in range(int(num_bins)):
        mask = bin_index == index
        count = int(np.count_nonzero(mask))
        centers_out.append(float(0.5 * (edges[index] + edges[index + 1])))
        counts_out.append(count)
        rms_out.append(
            _weighted_rms(sel_values[mask], sel_measures[mask])
            if count > 0
            else math.nan
        )

    bin_centers = np.asarray(centers_out, dtype=np.float64)
    bin_rms = np.asarray(rms_out, dtype=np.float64)
    bin_counts = np.asarray(counts_out, dtype=np.int64)

    fit_mask = (
        (bin_centers >= float(fit_min_distance))
        & (bin_centers <= float(fit_max_distance))
        & (bin_counts >= int(min_cells_per_bin))
        & np.isfinite(bin_rms)
        & (bin_rms > 0.0)
    )
    num_used = int(np.count_nonzero(fit_mask))
    if num_used < 3:
        raise MirrorDiagnosticsError(
            f"Only {num_used} usable bins in the fit window; at least 3 are required."
        )

    log_x = np.log(bin_centers[fit_mask])
    log_y = np.log(bin_rms[fit_mask])

    design = np.vstack([log_x, np.ones_like(log_x)]).T
    coefficients, residuals, _, _ = np.linalg.lstsq(design, log_y, rcond=None)
    slope = float(coefficients[0])

    predicted = design @ coefficients
    ss_residual = float(np.sum((log_y - predicted) ** 2))
    ss_total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0.0 else math.nan

    degrees_of_freedom = num_used - 2
    if degrees_of_freedom > 0 and ss_residual > 0.0:
        variance = ss_residual / degrees_of_freedom
        centred = log_x - np.mean(log_x)
        denominator = float(np.sum(centred * centred))
        stderr = math.sqrt(variance / denominator) if denominator > 0.0 else math.nan
    else:
        stderr = math.nan

    # Floor-aware refit over every usable bin, not only the fit window: the
    # innermost bins are precisely the ones that constrain the floor.
    usable = (
        (bin_counts >= int(min_cells_per_bin))
        & np.isfinite(bin_rms)
        & (bin_rms > 0.0)
        & (bin_centers > 0.0)
    )
    floor_fit = _fit_power_law_with_floor(
        bin_centers[usable], bin_rms[usable], initial_exponent=slope
    )

    return TransverseDecayMetrics(
        fit_min_distance=float(fit_min_distance),
        fit_max_distance=float(fit_max_distance),
        num_bins_used=num_used,
        decay_exponent=slope,
        decay_exponent_stderr=float(stderr),
        fit_r_squared=float(r_squared),
        floor_corrected_exponent=floor_fit["exponent"],
        floor_corrected_exponent_stderr=floor_fit["exponent_stderr"],
        floor_amplitude=floor_fit["floor"],
        floor_to_peak_ratio=floor_fit["floor_to_peak_ratio"],
        floor_model_converged=floor_fit["converged"],
        bin_centers=tuple(float(v) for v in bin_centers),
        bin_rms=tuple(float(v) for v in bin_rms),
        bin_cell_counts=tuple(int(v) for v in bin_counts),
    )


def _fit_power_law_with_floor(
    distances: np.ndarray,
    amplitudes: np.ndarray,
    *,
    initial_exponent: float,
) -> dict[str, Any]:
    """Fit rms(s)^2 = A^2 s^(2p) + F^2, returning p and the floor F.

    The fit is performed on ``log(rms)`` so that the several decades spanned by
    the profile are weighted evenly.  Failure to converge is reported rather
    than raised, so that one badly behaved map does not abort a whole sweep.
    """

    failed = {
        "exponent": math.nan,
        "exponent_stderr": math.nan,
        "floor": math.nan,
        "floor_to_peak_ratio": math.nan,
        "converged": False,
    }
    if distances.size < 4:
        return failed

    from scipy.optimize import curve_fit

    # Amplitudes span many decades, so the model is evaluated through a
    # log-sum-exp that cannot overflow while the optimiser explores.
    def model(s, log_amplitude, exponent, log_floor):
        log_power = 2.0 * log_amplitude + 2.0 * exponent * np.log(s)
        log_floor_term = np.full_like(log_power, 2.0 * log_floor)
        stacked = np.vstack([log_power, log_floor_term])
        return 0.5 * np.logaddexp(stacked[0], stacked[1])

    peak = float(np.max(amplitudes))
    reference = float(distances[np.argmax(amplitudes)])
    guess_exponent = float(initial_exponent) if math.isfinite(initial_exponent) else 2.0
    guess_exponent = float(np.clip(guess_exponent, 0.0, 6.0))
    guess = [
        math.log(peak / max(reference**guess_exponent, 1e-300)),
        guess_exponent,
        math.log(max(float(np.min(amplitudes)), peak * 1e-12)),
    ]

    log_peak = math.log(peak)
    try:
        parameters, covariance = curve_fit(
            model,
            distances,
            np.log(amplitudes),
            p0=guess,
            bounds=(
                [log_peak - 60.0, -2.0, log_peak - 60.0],
                [log_peak + 60.0, 8.0, log_peak + 1.0],
            ),
            maxfev=40000,
        )
    except (RuntimeError, ValueError, TypeError):
        return failed

    exponent = float(parameters[1])
    floor = float(np.exp(parameters[2]))
    stderr = (
        float(np.sqrt(covariance[1, 1]))
        if np.all(np.isfinite(covariance))
        else math.nan
    )
    floor_to_peak = floor / peak if peak > 0.0 else math.nan

    # When the fitted floor accounts for essentially the whole profile there is
    # no power-law component left to measure and the exponent is unidentifiable:
    # the optimiser returns whatever value the flat residual happens to favour.
    # This is the situation for a genuinely flat map, where the plain log-log
    # slope is the meaningful quantity, so the exponent is reported as missing
    # rather than as a spurious number.
    identifiable = math.isfinite(floor_to_peak) and floor_to_peak < FLOOR_DOMINANCE_LIMIT
    if not identifiable:
        return {
            "exponent": math.nan,
            "exponent_stderr": math.nan,
            "floor": floor,
            "floor_to_peak_ratio": floor_to_peak,
            "converged": False,
        }

    return {
        "exponent": exponent,
        "exponent_stderr": stderr,
        "floor": floor,
        "floor_to_peak_ratio": floor_to_peak,
        "converged": bool(math.isfinite(exponent) and math.isfinite(floor)),
    }


@dataclass(frozen=True)
class FloorExtrapolation:
    """Exponent extrapolated to a vanishing numerical floor."""

    num_points: int
    exponent_at_zero_floor: float
    exponent_at_zero_floor_stderr: float
    slope: float
    r_squared: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extrapolate_exponent_to_zero_floor(
    floor_to_peak_ratios: np.ndarray,
    exponents: np.ndarray,
) -> FloorExtrapolation:
    """Regress the measured exponent on the numerical floor and extrapolate.

    Each map carries its own discretisation floor, which depresses the measured
    exponent below the true one.  Across maps the floor varies by more than a
    decade, so regressing the exponent on the floor and reading the intercept
    estimates the exponent a floorless computation would produce.  The theorem
    predicts an intercept of 2.

    This is a diagnostic extrapolation, not a convergence study: it assumes the
    bias is affine in the floor over the observed range, which the fit quality
    can support but not prove.
    """

    ratios = np.asarray(floor_to_peak_ratios, dtype=np.float64)
    values = np.asarray(exponents, dtype=np.float64)

    if ratios.shape != values.shape:
        raise MirrorDiagnosticsError("Ratios and exponents must have the same shape.")

    finite = np.isfinite(ratios) & np.isfinite(values)
    ratios = ratios[finite]
    values = values[finite]
    if ratios.size < 3:
        raise MirrorDiagnosticsError(
            "At least three usable maps are required to extrapolate."
        )

    design = np.vstack([ratios, np.ones_like(ratios)]).T
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    predicted = design @ coefficients

    residual = values - predicted
    ss_residual = float(np.sum(residual * residual))
    ss_total = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0.0 else math.nan

    degrees_of_freedom = ratios.size - 2
    centred = ratios - np.mean(ratios)
    denominator = float(np.sum(centred * centred))
    if degrees_of_freedom > 0 and denominator > 0.0:
        variance = ss_residual / degrees_of_freedom
        stderr = math.sqrt(
            variance * float(np.sum(ratios * ratios)) / (ratios.size * denominator)
        )
    else:
        stderr = math.nan

    return FloorExtrapolation(
        num_points=int(ratios.size),
        exponent_at_zero_floor=float(coefficients[1]),
        exponent_at_zero_floor_stderr=float(stderr),
        slope=float(coefficients[0]),
        r_squared=float(r_squared),
    )


def mirror_orientation_residual(
    dominant_mode: int,
    orientation_degrees: float,
) -> float:
    """Angular distance to the nearest mirror-compatible orientation.

    A single angular mode ``cos(n (theta - phi))`` is even under the mirror
    ``theta -> pi - theta`` if and only if

        phi = 90 degrees   modulo   180 / n degrees.

    The returned residual is the unsigned distance, in degrees, between the
    fitted orientation and that admissible set.  The theorem predicts a residual
    of the order of the angular discretisation, for every modality.
    """

    if dominant_mode < 1:
        raise MirrorDiagnosticsError("dominant_mode must be a positive integer.")

    period = 180.0 / float(dominant_mode)
    offset = (float(orientation_degrees) - 90.0) % period
    return float(min(offset, period - offset))
