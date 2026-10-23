#!/usr/bin/env python3
"""Confront the measured u_ref strain profiles with the measured kernel drops.

``measure_normal_strain_axis_profile.py`` produces, from the reference field
u_ref alone, the profile across the mirror axis of the time-integrated squared
normal strains.  u_ref lives in the healthy medium and does not depend on the
defect, so those profiles are known without reference to any off-axis campaign.

This script evaluates each profile at the actual off-axis displacement used by
the campaign (1.0 length unit, i.e. four defect radii) and compares the
resulting prediction with the target-zone kernel drops measured by
``compare_off_axis_absolute_amplitudes.py``.

Two predictions are formed, bracketing the unknown behaviour of the adjoint:

  * ``u_ref only``  -- only the u_ref factor of the kernel moves; the prediction
    is the amplitude ratio, i.e. the square root of the profile ratio.  This is
    a lower bound on the drop (an upper bound on the surviving fraction).
  * ``both factors`` -- the adjoint's normal strain follows the same profile as
    u_ref's; the prediction is the profile ratio itself.

The classical, reflection-only and transmission-only kernels all pair u_ref, so
they are the ones this comparison applies to.  The residual-interaction kernel
never pairs u_ref and is reported for contrast only, not as a test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

U_REF_PAIRING = ("classical_full", "standard_reflection_only", "standard_transmission_only")


def interpolate(profile: dict, s_query: float) -> float:
    s = np.asarray(profile["s"], dtype=float)
    values = np.asarray(profile["time_integrated_square"], dtype=float)
    order = np.argsort(s)
    return float(np.interp(s_query, s[order], values[order]))


def symmetric_value(profile: dict, s_query: float) -> float:
    """Average the profile at +s and -s: parity says they are equal."""
    return 0.5 * (interpolate(profile, s_query) + interpolate(profile, -s_query))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--amplitudes-csv", type=Path, required=True)
    parser.add_argument("--shift", type=float, default=1.0, help="off-axis displacement of the campaign")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    profiles = json.loads(args.profile_json.read_text(encoding="utf-8"))
    measured = {
        (row["combination"], row["parameter"]): float(row["target_peak_ratio"])
        for row in csv.DictReader(args.amplitudes_csv.open(encoding="utf-8"))
    }

    result: dict[str, object] = {
        "kind": "strain_profile_versus_kernel_drop",
        "shift": args.shift,
        "profile_json": str(args.profile_json),
        "amplitudes_csv": str(args.amplitudes_csv),
        "parameters": {},
    }

    print(f"Off-axis displacement s = {args.shift}\n")
    for parameter, key in (("C11", "profile_eps11"), ("C22", "profile_eps22")):
        on_axis = symmetric_value(profiles[key], 0.0)
        shifted = symmetric_value(profiles[key], args.shift)
        profile_ratio = shifted / on_axis
        prediction_both = profile_ratio
        prediction_uref_only = float(np.sqrt(profile_ratio))

        entry = {
            "time_integrated_square_on_axis": on_axis,
            "time_integrated_square_at_shift": shifted,
            "profile_ratio": profile_ratio,
            "prediction_u_ref_only": prediction_uref_only,
            "prediction_both_factors": prediction_both,
            "measured": {},
        }
        print(f"--- {parameter} ---")
        print(f"  profil u_ref: {on_axis:.5e} -> {shifted:.5e}   ratio = {profile_ratio:.4f}")
        print(f"  prediction 'u_ref seul'    : x{prediction_uref_only:.3f}")
        print(f"  prediction 'les 2 facteurs': x{prediction_both:.3f}")
        for combination in U_REF_PAIRING:
            value = measured.get((combination, parameter))
            if value is None:
                continue
            entry["measured"][combination] = value
            inside = min(prediction_both, prediction_uref_only) * 0.9 <= value <= max(prediction_both, prediction_uref_only) * 1.1
            flag = "dans la fourchette" if inside else "HORS fourchette"
            print(f"    mesure {combination:26s} x{value:.3f}   {flag}")
        ri = measured.get(("residual_interaction", parameter))
        if ri is not None:
            entry["residual_interaction_for_contrast"] = ri
            print(f"    (contraste) residual_interaction x{ri:.3f}  -- n'apparie pas u_ref, hors test")
        result["parameters"][parameter] = entry
        print()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
