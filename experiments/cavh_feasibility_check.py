"""experiments/cavh_feasibility_check.py -- commanded amplitude against CAV-H aerodynamic limits.

trajectory_generators/profiles.py commands lift as a free control input with no dynamic-pressure
check, and tabulates the gap against a sourced CAV-H model: supersonic_cruise's 13.2 g needs
m/(S*CL) = 402 kg/m^2, 4.7x the CAV-H value. The paper scales the command by a feasibility factor,
and Section 2 gives three readings of that factor.

All three readings are computed by sim/sixdof.a_max_at(alt, mach, DEFAULT_AIRFRAME), and
DEFAULT_AIRFRAME is a 200 kg tactical interceptor, a different vehicle from the hypersonic cruise
vehicle that flies the trajectory. If the interceptor has the higher lift capability, a factor
derived from it is loose by the capability ratio.

The script computes, from the project's own models:

  1. Lift loading m/(S*CL_max) for the interceptor and for CAV-H, and their ratio.
  2. Self-consistency of the CAV-H model in profiles.py: the loading from CAVH_MASS, CAVH_S and
     CL_max of the Xu-Hu-Pan polynomial at the stall angle, compared with the loading implied by
     the tabulated 4.7x (402 kg/m^2 x 4.7).
  3. Per flown trajectory: the commanded g, the g the interceptor can pull there, and the g CAV-H
     can pull there, for the reported factor 0.2798 and the alpha-limited factor 1/4.7.

Acceptance criterion: the reported amplitude is CAV-H-feasible if it lies under CAV-H's limit on at
least 90% of flown trajectories. Otherwise the script reports the factor that is CAV-H-feasible at
the median condition. The alpha-limited reading in Section 2, 1/4.7, equals the CAV-H ratio.

trajectory_generators/ is used unmodified; the amplitude factor is applied at the measurement layer.

    python experiments/cavh_feasibility_check.py --json runs/ml/cavh_feasibility.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_generators.profiles import cavh_CL, CAVH_S, CAVH_MASS        # noqa: E402
from sim.sixdof import DEFAULT_AIRFRAME, a_max_at, sound_speed, G0           # noqa: E402
from trajectory_generators.atmosphere import density                          # noqa: E402

AMP = 0.2798            # the reported feasibility factor
ALT_AMP = 1.0 / 4.7     # the alpha-limited alternative in Section 2
SEEDS = 30
STALL_DEG = 20.0        # the airframe's alpha bound, aoa_max_deg(DEFAULT_AIRFRAME)
REF_MACH = 5.5          # Mach of the reference condition in Section 2 (25.2 km, Mach 5.5)


def cavh_a_max(alt_m, mach, alpha_deg=STALL_DEG):
    """Lateral acceleration (m/s^2) CAV-H can pull at (alt_m, mach) and alpha_deg, from the CL
    polynomial in profiles.py."""
    V = mach * sound_speed(max(alt_m, 0.0))
    qbar = 0.5 * density(max(alt_m, 0.0)) * V * V
    return qbar * CAVH_S * cavh_CL(alpha_deg, mach) / CAVH_MASS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    print("CAV-H FEASIBILITY CHECK: commanded amplitude against the lift limit of the flying vehicle\n")

    # ---- 1. the two vehicles, side by side --------------------------------------------------
    # CL_max is evaluated at the reference condition of the paper, 25.2 km and Mach 5.5
    # (CAV-H loading 1901.1 kg/m^2; 1918.0 kg/m^2 at Mach 6.0).
    cl_max = cavh_CL(STALL_DEG, REF_MACH)
    cavh_load = CAVH_MASS / (CAVH_S * cl_max)
    af = DEFAULT_AIRFRAME
    int_load = af["m"] / (af["Sref"] * af["CN_max"])
    print("  1. LIFT LOADING m/(S*CL_max), the quantity that sets available g at a given qbar")
    print(f"     interceptor (DEFAULT_AIRFRAME)  {int_load:8.1f} kg/m^2   "
          f"(m={af['m']:.0f}, S={af['Sref']:.4f}, CN_max={af['CN_max']:.1f})")
    print(f"     CAV-H (profiles.py's own model) {cavh_load:8.1f} kg/m^2   "
          f"(m={CAVH_MASS:.0f}, S={CAVH_S:.4f}, CL_max={cl_max:.3f} at {STALL_DEG:.0f} deg, M6)")
    print(f"     the interceptor is {cavh_load/int_load:.2f}x more capable per unit dynamic pressure\n")

    # ---- 2. consistency of the tabulated 4.7x with the CAV-H polynomial ------------------------
    # profiles.py tabulates supersonic_cruise's 13.2 g as needing m/(S*CL) = 402 kg/m^2, 4.7x a
    # sourced CAV-H. 402 x 4.7 is compared with the loading from the polynomial; agreement within
    # 10% counts as consistent.
    implied = 402.0 * 4.7
    err = abs(implied - cavh_load) / cavh_load
    print("  2. CONSISTENCY OF THE TABULATED 4.7x WITH THE CAV-H MODEL")
    print(f"     from the table        402 kg/m^2 x 4.7 = {implied:.0f} kg/m^2")
    print(f"     from the polynomial                     {cavh_load:.0f} kg/m^2")
    print(f"     disagreement {100*err:.1f}%  -> {'CONSISTENT' if err < 0.10 else '*** INCONSISTENT ***'}\n")

    # ---- 3. per flown trajectory ---------------------------------------------------------------
    from experiments.multiclass_lead import class_windows
    rows = []
    for sd in range(args.seeds):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        for w in wins:
            raw_g = float(np.max(np.asarray(w["a_cmd"], float))) / G0     # generator's own command
            alt, mach = float(w["alt"]), float(w["mach"])
            rows.append(dict(alt_km=alt / 1e3, mach=mach, raw_cmd_g=raw_g,
                             shipped_cmd_g=raw_g * AMP, alt_cmd_g=raw_g * ALT_AMP,
                             interceptor_max_g=a_max_at(alt, mach) / G0,
                             cavh_max_g=cavh_a_max(alt, mach) / G0))
    if not rows:
        print("  no trajectories formed"); return

    g = lambda k: np.array([r[k] for r in rows], float)                       # noqa: E731
    ship_ok = g("shipped_cmd_g") <= g("cavh_max_g")
    alt_ok = g("alt_cmd_g") <= g("cavh_max_g")
    int_ok = g("shipped_cmd_g") <= g("interceptor_max_g")
    print(f"  3. PER FLOWN TRAJECTORY (n={len(rows)}), median values")
    print(f"     generator's raw command      {np.median(g('raw_cmd_g')):6.2f} g")
    print(f"     reported (x{AMP:.4f})           {np.median(g('shipped_cmd_g')):6.2f} g")
    print(f"     alpha-limited (x{ALT_AMP:.4f})     {np.median(g('alt_cmd_g')):6.2f} g")
    print(f"     interceptor can pull         {np.median(g('interceptor_max_g')):6.2f} g")
    print(f"     CAV-H can pull               {np.median(g('cavh_max_g')):6.2f} g\n")
    print(f"     reported amplitude under CAV-H's limit     : {int(ship_ok.sum())}/{len(rows)}")
    print(f"     alpha-limited under CAV-H's limit          : {int(alt_ok.sum())}/{len(rows)}")
    print(f"     reported amplitude under interceptor's limit: {int(int_ok.sum())}/{len(rows)}")

    frac = float(ship_ok.mean())
    verdict = frac >= 0.90
    print(f"\n  RESULT   reported amplitude is CAV-H-feasible on {100*frac:.0f}% "
          f"(bar 90%) -> {'feasible' if verdict else 'not feasible for CAV-H'}")
    if not verdict:
        need = float(np.median(g("cavh_max_g") / g("raw_cmd_g")))
        print(f"  the factor that is CAV-H-feasible at the median condition: {need:.4f} "
              f"(reported {AMP:.4f}, alpha-limited {ALT_AMP:.4f})")

    out = dict(seeds=args.seeds, amp_shipped=AMP, amp_alpha_limited=ALT_AMP,
               stall_deg=STALL_DEG, interceptor_load=int_load, cavh_load=cavh_load,
               capability_ratio=cavh_load / int_load, cavh_cl_max=cl_max,
               disclosure_implied_load=implied, disclosure_error=err,
               disclosure_consistent=bool(err < 0.10),
               n=len(rows), frac_shipped_cavh_feasible=frac,
               frac_alpha_cavh_feasible=float(alt_ok.mean()),
               frac_shipped_interceptor_feasible=float(int_ok.mean()),
               shipped_is_cavh_feasible=bool(verdict),
               cavh_feasible_factor_median=float(np.median(g("cavh_max_g") / g("raw_cmd_g"))),
               rows=rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
