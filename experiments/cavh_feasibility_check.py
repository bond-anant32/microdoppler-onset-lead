"""experiments/cavh_feasibility_check.py -- WHOSE aerodynamics decides the commanded amplitude?

THE SUSPICION. profiles.py discloses that the generators command lift as a free control input with
no dynamic-pressure check, and tabulates the gap against a sourced CAV-H: supersonic_cruise's
13.2 g needs m/(S*CL) = 402 kg/m^2, which is 4.7x what CAV-H has. The letter's answer is to scale
the command down by a feasibility factor, and Section II carries three defensible readings of it.

But every one of those readings is computed by sim/sixdof.a_max_at(alt, mach, DEFAULT_AIRFRAME) --
and DEFAULT_AIRFRAME is a 200 kg tactical INTERCEPTOR, not the hypersonic cruise vehicle whose
trajectory is being flown. So the vehicle that flies the trajectory and the vehicle whose
aerodynamics decides what it may command are two different vehicles. If the interceptor is the
more capable of the two, the feasibility gate is loose by exactly that ratio and the "feasible"
amplitude is not feasible.

WHAT THIS MEASURES, all of it from the project's own numbers, none of it asserted:

  1. The two vehicles' lift loading m/(S*CL_max), interceptor against CAV-H, and their ratio.
  2. Whether trajectory_generators/profiles.py's OWN CAV-H model is self-consistent -- CL_max from
     the Xu-Hu-Pan polynomial at the stall angle against the m/(S*CL_max) implied by the disclosed
     4.7x, and against CAVH_MASS/CAVH_S. Two independent routes to the same number, or the
     "sourced CAV-H" column is not sourced.
  3. Per FLOWN trajectory: the commanded g, the g the interceptor could pull there, and the g
     CAV-H could pull there. The question is whether the shipped 0.2798 command sits under the
     CAV-H curve or only under the interceptor's.

ACCEPTANCE, stated before running:
  - if the shipped amplitude is under CAV-H's own limit on >= 90% of flown trajectories, the
    feasibility factor is sound and the interceptor's aero was a harmless stand-in;
  - if it is not, the letter is flying a hypersonic vehicle at an amplitude only an interceptor
    could sustain, and the amplitude axis has to be re-anchored on the vehicle that flies.
Either outcome is reportable, and the second one is fixable -- Section II already carries an
alpha-limited reading at 1/4.7, which is the CAV-H ratio.

NOTHING IN trajectory_generators/ IS EDITED. Those twelve files are byte-identical to Researchzip
(verified by md5), the dataset of a different paper depends on them, and profiles.py
states the non-correction is deliberate. The fix, if one is needed, belongs where the letter
already puts it: the amplitude factor at the measurement layer.

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

AMP = 0.2798            # the shipped feasibility factor
ALT_AMP = 1.0 / 4.7     # the alpha-limited alternative Sec. II also carries
SEEDS = 30
STALL_DEG = 20.0        # the airframe's own alpha bound, aoa_max_deg(DEFAULT_AIRFRAME)
REF_MACH = 5.5          # the reference condition Sec. II states, and the Mach
                        # experiments/aero_feasible_factor.py uses. The two compute the same
                        # lift loading, so they must be evaluated at the same Mach.


def cavh_a_max(alt_m, mach, alpha_deg=STALL_DEG):
    """Lateral g CAV-H can pull at (alt, Mach), from profiles.py's OWN polynomial."""
    V = mach * sound_speed(max(alt_m, 0.0))
    qbar = 0.5 * density(max(alt_m, 0.0)) * V * V
    return qbar * CAVH_S * cavh_CL(alpha_deg, mach) / CAVH_MASS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    print("CAV-H FEASIBILITY CHECK -- is the flown amplitude feasible for the vehicle that flies?\n")

    # ---- 1. the two vehicles, side by side --------------------------------------------------
    # MACH IS THE LETTER'S STATED REFERENCE CONDITION, NOT 6.0. The manuscript sentence that
    # quotes this number opens "At 25.2 km and Mach 5.5", and experiments/aero_feasible_factor.py
    # computes the same quantity at its own MACH = 5.5. Both producers must evaluate at the stated
    # condition, or one number has two values (1901.1 against 1918.0 at Mach 6.0).
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

    # ---- 2. is profiles.py's CAV-H self-consistent with its own disclosed 4.7x? ---------------
    # The disclosure says supersonic_cruise's 13.2 g "needs m/(S*CL) = 402 kg/m^2 ... 4.7x a
    # sourced CAV-H". Two independent routes to CAV-H's loading must agree, or that column is a
    # remembered number rather than a computed one.
    implied = 402.0 * 4.7
    err = abs(implied - cavh_load) / cavh_load
    print("  2. IS THE DISCLOSED 4.7x CONSISTENT WITH THE SHIPPED CAV-H MODEL?")
    print(f"     from the disclosure   402 kg/m^2 x 4.7 = {implied:.0f} kg/m^2")
    print(f"     from the polynomial                     {cavh_load:.0f} kg/m^2")
    print(f"     disagreement {100*err:.1f}%  -> {'CONSISTENT' if err < 0.10 else '*** INCONSISTENT ***'}\n")

    # ---- 3. per FLOWN trajectory ---------------------------------------------------------------
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
    print(f"     shipped (x{AMP:.4f})           {np.median(g('shipped_cmd_g')):6.2f} g")
    print(f"     alpha-limited (x{ALT_AMP:.4f})     {np.median(g('alt_cmd_g')):6.2f} g")
    print(f"     interceptor can pull         {np.median(g('interceptor_max_g')):6.2f} g")
    print(f"     CAV-H can pull               {np.median(g('cavh_max_g')):6.2f} g\n")
    print(f"     shipped amplitude under CAV-H's limit      : {int(ship_ok.sum())}/{len(rows)}")
    print(f"     alpha-limited under CAV-H's limit          : {int(alt_ok.sum())}/{len(rows)}")
    print(f"     shipped amplitude under interceptor's limit: {int(int_ok.sum())}/{len(rows)}")

    frac = float(ship_ok.mean())
    verdict = frac >= 0.90
    print(f"\n  VERDICT  shipped amplitude is CAV-H-feasible on {100*frac:.0f}% "
          f"(bar 90%) -> {'SOUND' if verdict else '*** NOT FEASIBLE FOR THE VEHICLE THAT FLIES ***'}")
    if not verdict:
        need = float(np.median(g("cavh_max_g") / g("raw_cmd_g")))
        print(f"  the factor that WOULD be CAV-H-feasible at the median condition: {need:.4f} "
              f"(shipped {AMP:.4f}, alpha-limited {ALT_AMP:.4f})")

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
