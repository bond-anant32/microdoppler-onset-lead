"""experiments/aero_feasible_sweep.py -- does the lead survive an AERODYNAMICALLY FEASIBLE maneuver?

THE PROBLEM THIS TESTS. supersonic_cruise flies 13.2 g at 25.2 km / M5.5, where qbar = 52.0 kPa. Producing that lift needs
m/(S*CL_max) = 402 kg/m^2 -- 14.3x the evader model's capability and 4.7x a SOURCED CAV-H
wind-tunnel fit. The gated long-corridor configuration is worse: 15.6 g at 27.3 km, 21.3x
and 7.1x respectively. The amplitude is an authored constant (weave_lat_accel = 130.0 m/s^2), not a
physics result, and the trajectory gate cannot catch it because its g_cap carries no
dynamic-pressure term.

So the one class on which the lead is measured may be flying a maneuver the air cannot supply. That
threatens the headline directly, because the detection statistic is the trailing mean of
|dphi/dt|^2 and |dphi/dt| goes as FIN RATE: scaling the command down ~4.7x could cost ~22x in the
statistic, and a cue that no longer clears its own noise floor has no lead at all.

WHY THE ANSWER IS NOT OBVIOUS, i.e. why this is measured rather than argued. Two effects oppose:

  (+) The onset threshold (2.0 m/s^2) is FIXED. A smaller command takes LONGER to cross it, so the
      budget t_on - t_c GROWS. Since a_z rises far more slowly than delta -- the fin is at 80 % of
      its excursion while |a_z| is still ~2 % -- t_on should move out MORE than the muD alarm does.
      On that reasoning the over-driven case UNDERSTATES the advantage.
  (-) The statistic goes as rate SQUARED. Below some amplitude the fin transient stops clearing the
      cruise-maximum threshold and detection collapses to zero, at which point there is no lead.

Which dominates is empirical. This sweeps the commanded amplitude over the feasibility factors the
aero audit actually reports and measures budget, detection, no-cue false alarms, and the paired
advantage against each kinematic arm at every point.

    <PY> experiments/aero_feasible_sweep.py --seeds 30 --json runs/ml/aero_feasible.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from scipy.stats import wilcoxon                                             # noqa: E402
from experiments.multiclass_lead import class_windows, measure                # noqa: E402

# 1.0 is the dataset amplitude.
#
# THE REPORTED ROW IS THE DERIVED ONE, 0.2798 -- the factor experiments/aero_feasible_factor.py
# derives and the factor the letter says it flies. The alpha-limited value at 1/4.7 is kept
# alongside it as the sensitivity the letter states in Sec. II ("at which every advantage roughly
# doubles"), so that claim is measurable in-tree instead of asserted.
#
# THE PER-TRAJECTORY ROW. The derived factor is ONE scalar, evaluated at the aerodynamic
# reference condition of 25.2 km / M5.50 where qbar = 52.4 kPa. The supersonic-cruise
# trajectories are flown at a per-trajectory MEDIAN qbar of 86.9 kPa (mean 94.9; qbar at the mean
# 24.6 km / M6.84 condition is 88.8) -- 1.66x the 52.4 kPa reference. The letter prints the median
# with its construction named, rather than a single tuning-point value. Either way the
# air can supply far more lateral acceleration than the reference condition suggests. Computing
# a_avail/a_cmd per trajectory from the project's own atmosphere and CAV-H polar gives a median
# factor of 0.4931, and 28 of 30 trajectories exceed the 0.2798 that is flown.
#
# That direction matters and it is not the flattering one: the letter commands a GENTLER maneuver
# than its own feasibility rule permits at the conditions it flies, and the advantage rises as
# commanded g falls. So the fixed scalar INFLATES the headline relative to a per-trajectory gate.
# "We take the less favourable of the two" was true of 1/3.57 vs 1/4.7 and false of the choice
# that actually matters. This row replaces the arbitrary "half" so the comparison is measured.
DERIVED_FACTOR = 0.2798            # aero_feasible_factor.py, C_L,max = 1.296 at alpha 25 deg
PER_TRAJ_FACTOR = 0.4931           # median a_avail/a_cmd at the flown conditions
REPORTED_LABEL = "CAV-H feasible, derived CLmax"
FACTORS = [
    (1.0,             "as shipped (13.2-15.6 g)"),
    (PER_TRAJ_FACTOR, "per-trajectory feasible (median)"),
    (DERIVED_FACTOR, REPORTED_LABEL),
    (1 / 4.7,        "CAV-H feasible, alpha-limited CLmax"),
    (1 / 7.1,        "CAV-H feasible, gated long cfg"),
    (1 / 14.3,       "evader-model feasible"),
    (2.0,            "double (does the lead grow with g?)"),
]


def boot(v, seed=11, n=8000):
    r = np.random.default_rng(seed)
    v = np.asarray(v, float)
    if v.size < 3:
        return float("nan"), float("nan")
    m = [np.median(r.choice(v, v.size, replace=True)) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--kin-noise", type=float, default=None,
                    help="comparator measurement-noise sigma (m/s^2); default 0.3")
    ap.add_argument("--dwell", type=float, default=0.002)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("AERODYNAMIC FEASIBILITY SWEEP -- supersonic_cruise, %d seeds\n" % args.seeds)
    print("  The weave amplitude is 4.7-7.1x a sourced CAV-H wind-tunnel fit.")
    print("  Scaling the COMMAND scales fin deflection and hence fin rate; the statistic goes as")
    print("  rate^2, while the 2.0 m/s^2 onset threshold and the noise floor stay fixed.\n")
    print("%-32s %8s %8s %6s %6s %22s %20s"
          % ("commanded amplitude", "mean g", "budget", "det", "FA", "adv vs CUSUM (ms)",
             "adv vs GLR (ms)"))
    print("-" * 118)

    out = []
    for fac, lab in FACTORS:
        # Mirrors sweep_class(): one trajectory per seed, events pooled by median WITHIN a
        # trajectory, statistics formed ACROSS trajectories. n counts trajectories, not events.
        per_arm, budgets, dets, fas, gs = {}, [], [], [], []
        n_entered = n_no_window = n_mu_never = 0
        for sd in range(args.seeds):
            try:
                wins, _d = class_windows("supersonic_cruise",
                                         rng=np.random.default_rng(90000 + sd),
                                         amp_factor=fac)
            except Exception:                                                 # noqa: BLE001
                continue
            if not wins:
                n_no_window += 1
                continue
            n_entered += 1
            arm_ev, buds, gg = {}, [], []
            for w in wins:
                w2 = dict(w)
                w2["a_cmd"] = np.asarray(w["a_cmd"], float) * fac
                try:
                    m = measure(w2, args.snr, args.reps, args.dwell, kin_noise=args.kin_noise)
                except Exception:                                             # noqa: BLE001
                    m = None
                if not m:
                    continue
                buds.append(m["budget_ms"]); dets.append(m["det"]); fas.append(m["fa"])
                gg.append(w["amp_g"] * fac)
                for arm, st in (m.get("arms") or {}).items():
                    if st.get("adv") is not None:
                        arm_ev.setdefault(arm, []).append(st["adv"])
            if buds:
                budgets.append(float(np.median(buds)))
                gs.append(float(np.median(gg)))
            if buds and not arm_ev.get("CUSUM Page54"):
                # the muD arm never alarmed on this trajectory: the pair never forms and the row
                # silently vanishes. That is exactly the case where muD LOST, so it must be counted.
                n_mu_never += 1
            for k, v in arm_ev.items():
                per_arm.setdefault(k, []).append(float(np.median(v)))

        if not budgets:
            print("%-32s %8s %8s %6s %6s   no measurable onset" % (lab, "-", "-", "-", "-"))
            out.append(dict(factor=fac, label=lab, n=0))
            continue

        cells = {}
        for arm in ("CUSUM Page54", "GLR Willsky76"):
            v = np.asarray(per_arm.get(arm, []), float)
            if v.size >= 3:
                lo, hi = boot(v)
                p = wilcoxon(v).pvalue if np.any(v != 0) else float("nan")
                cells[arm] = "%+.1f [%+.1f,%+.1f] n=%d" % (np.median(v), lo, hi, v.size)
                out_p = p
            else:
                cells[arm] = "n=%d (too few)" % v.size
                out_p = float("nan")
            per_arm[arm + "_p"] = out_p
        print("%-32s %8.2f %8.1f %5.0f%% %5.0f%% %22s %20s"
              % (lab, float(np.mean(gs)), float(np.mean(budgets)),
                 100 * np.mean(dets), 100 * np.mean(fas),
                 cells["CUSUM Page54"], cells["GLR Willsky76"]))
        out.append(dict(factor=fac, label=lab, n=len(budgets),
                        seeds=args.seeds, snr_db=args.snr, reps=args.reps,
                        kin_noise=(0.3 if args.kin_noise is None else args.kin_noise),
                        dwell_s=args.dwell, amp_factor=fac,
                        n_entered=n_entered, n_no_window=n_no_window,
                        n_mu_never_alarmed=n_mu_never,
                        mean_g=float(np.mean(gs)), mean_budget_ms=float(np.mean(budgets)),
                        det=float(np.mean(dets)), fa=float(np.mean(fas)),
                        adv={k: list(map(float, v)) for k, v in per_arm.items()
                             if isinstance(v, list)}))

    print("\nREAD THIS COLUMN-WISE. If the advantage holds or GROWS as the amplitude falls to the")
    print("CAV-H-feasible rows, the headline is robust and gains a sourced feasibility argument.")
    print("If detection collapses there, the headline is an artifact of an over-driven maneuver")
    print("and must be restated at the feasible amplitude.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
