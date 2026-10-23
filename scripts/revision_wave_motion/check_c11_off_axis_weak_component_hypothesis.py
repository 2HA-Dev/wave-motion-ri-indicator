#!/usr/bin/env python3
"""Test whether the C11 off-axis exaltation tracks which field each kernel pairs.

Context. The off-axis campaign shows that moving the defect one unit off the mirror axis lets the C_66
sensitivity rise by 2-3 orders of magnitude, but also that the C_11 sensitivity
*falls* off axis, and does so by a different factor depending on which pair of
fields the kernel contracts:

    classical_full              x0.274
    standard_reflection_only    x0.284
    standard_transmission_only  x0.275
    residual_interaction        x0.539

Three constructions cluster tightly near x0.27-0.28; residual_interaction is
roughly twice as robust. For the polarisation-reversal campaign, a kernel's
numerical floor tracks the amplitude of the *weaker* of the two fields it
contracts, not the map itself. This script tests the same idea here, using
only data already on disk (receiver-array time series in
``inverse/vector_array``), with no PDE solve.

C_11 is built from the strain component epsilon_11 = d(u_1)/dx_1, i.e. from the
x-component of whichever fields a construction contracts.  ``classical_full``,
``standard_reflection_only`` and ``standard_transmission_only`` all contract the
healthy-medium reference field u_ref against an adjoint field; ``residual_interaction``
contracts the reflection residual W_Gamma1 against the transmission residual
Lambda_Gamma2 and never touches u_ref.  If u_ref's x-component is far weaker,
relative to its y-component, than the residual fields' x-components, the three
u_ref-based constructions should collapse together off axis while
residual_interaction should be comparatively spared, which is exactly the
clustering observed above.

This script measures the RMS(x)/RMS(y) ratio of u_ref (sampled at both receiver
arrays), the reflection residual W_Gamma1, and the transmission residual
Lambda_Gamma2, for the on-axis reference campaign and for the off-axis campaign,
and reports whether the ordering matches the prediction. It is a structural,
qualitative check (four data points do not support a fitted correlation),
not a proof.  Closing this analytically requires extending the symbolic proof
(``verify_mirror_symmetry_symbolic.py``) to the second-order term away from the
axis; that is not attempted here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rms_xy_ratio(path: Path) -> tuple[float, float, float]:
    data = np.load(path)
    values = data["values"]
    components = [str(c) for c in data["components"]]
    ix, iy = components.index("x"), components.index("y")
    rms = np.sqrt((values**2).mean(axis=(0, 1)))
    return float(rms[ix] / rms[iy]), float(rms[ix]), float(rms[iy])


def campaign_ratios(vector_array_dir: Path) -> dict[str, dict[str, float]]:
    fields = {
        "u_ref_at_gamma1": vector_array_dir / "simulated_reflection_signals.npz",
        "u_ref_at_gamma2": vector_array_dir / "simulated_transmission_signals.npz",
        "W_gamma1_reflection_residual": vector_array_dir / "vector_array_reflection_residuals.npz",
        "Lambda_gamma2_transmission_residual": vector_array_dir / "vector_array_transmission_residuals.npz",
    }
    out: dict[str, dict[str, float]] = {}
    for name, path in fields.items():
        ratio, rms_x, rms_y = rms_xy_ratio(path)
        out[name] = {"x_over_y_ratio": ratio, "rms_x": rms_x, "rms_y": rms_y}
    return out


# Measured off-axis / on-axis peak ratio for C11 in the target zone, taken
# from off_axis_absolute_amplitudes.csv (compare_off_axis_absolute_amplitudes.py).
MEASURED_C11_EXALTATION_DROP = {
    "classical_full": 0.2743611415637176,
    "standard_reflection_only": 0.284023982356852,
    "standard_transmission_only": 0.27494204020909147,
    "residual_interaction": 0.5388896388371283,
}
# Which fields each construction contracts to build the C11 kernel.
CONSTRUCTION_FIELDS = {
    "classical_full": ("u_ref_at_gamma1", "u_ref_at_gamma2"),
    "standard_reflection_only": ("u_ref_at_gamma1", "W_gamma1_reflection_residual"),
    "standard_transmission_only": ("u_ref_at_gamma2", "Lambda_gamma2_transmission_residual"),
    "residual_interaction": ("W_gamma1_reflection_residual", "Lambda_gamma2_transmission_residual"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--on-axis-vector-array-dir", type=Path, required=True)
    parser.add_argument("--off-axis-vector-array-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "kind": "c11_off_axis_weak_component_hypothesis_check",
        "on_axis_ratios": campaign_ratios(args.on_axis_vector_array_dir),
        "off_axis_ratios": campaign_ratios(args.off_axis_vector_array_dir),
    }

    print("RMS(x)/RMS(y), on axis -> off axis:")
    for name in result["on_axis_ratios"]:
        a = result["on_axis_ratios"][name]["x_over_y_ratio"]
        b = result["off_axis_ratios"][name]["x_over_y_ratio"]
        print(f"  {name:38s} {a:10.4e} -> {b:10.4e}")

    print("\nMeasured C11 off/on target-peak ratio, against which field pair each construction contracts:")
    rows = []
    for combination, drop in MEASURED_C11_EXALTATION_DROP.items():
        field_a, field_b = CONSTRUCTION_FIELDS[combination]
        weakest_link = min(
            result["on_axis_ratios"][field_a]["x_over_y_ratio"],
            result["on_axis_ratios"][field_b]["x_over_y_ratio"],
        )
        involves_u_ref = field_a.startswith("u_ref") or field_b.startswith("u_ref")
        rows.append(
            {
                "combination": combination,
                "measured_C11_off_over_on_ratio": drop,
                "paired_fields": [field_a, field_b],
                "weakest_link_x_over_y_ratio_on_axis": weakest_link,
                "involves_u_ref": involves_u_ref,
            }
        )
        print(
            f"  {combination:26s} drop={drop:6.3f}  weakest_link_ratio={weakest_link:10.4e}"
            f"  involves_u_ref={involves_u_ref}"
        )
    result["rows"] = rows

    predicted_order = sorted(rows, key=lambda r: r["weakest_link_x_over_y_ratio_on_axis"])
    observed_order = sorted(rows, key=lambda r: r["measured_C11_off_over_on_ratio"])
    prediction_holds = [r["combination"] for r in predicted_order] == [r["combination"] for r in observed_order]
    result["rank_order_matches_prediction"] = prediction_holds
    print(f"\nRank order (weakest link -> most fragile) matches measured drop order: {prediction_holds}")

    (args.output_dir / "c11_weak_component_hypothesis_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
