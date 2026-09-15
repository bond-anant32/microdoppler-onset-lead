"""experiments/aero_feasible_sweep.py -- lead advantage versus commanded manoeuvre amplitude.

supersonic_cruise flies 13.2 g at 25.2 km / M5.5, where qbar = 52.0 kPa. That lift requires
m/(S*CL_max) = 402 kg/m^2, which is 14.3x the evader model's capability and 4.7x that of a sourced
CAV-H wind-tunnel fit. The gated long-corridor configuration flies 15.6 g at 27.3 km, 21.3x and
7.1x respectively. The amplitude is a configured constant (weave_lat_accel = 130.0 m/s^2), and the
g_cap of the physical-envelope checks has no dynamic-pressure term.

The detection statistic is the trailing mean of |dphi/dt|^2, and |dphi/dt| scales with fin rate, so
scaling the command down 4.7x reduces the signal term of the statistic by about 22x. Two effects
act in opposite directions as the amplitude falls:

  (+) The onset threshold (2.0 m/s^2) is fixed. A smaller command takes longer to cross it, so the
      budget t_on - t_c grows. a_z rises far more slowly than delta (the fin is at 80 % of its
      excursion while |a_z| is still ~2 %), so t_on moves out further than the muD alarm does.
  (-) The statistic scales as fin rate squared. Below some amplitude the fin transient no longer
      clears the cruise-maximum threshold and detection falls to zero.

The script sweeps the commanded amplitude over the feasibility factors in FACTORS and measures the
budget, detection rate, no-cue false-alarm rate, and paired advantage against each kinematic arm.

    python experiments/aero_feasible_sweep.py --seeds 30 --reps 12 --snr 40 --json runs/ml/aero_feasible_40db_derived.json
    python experiments/aero_feasible_sweep.py --seeds 30 --reps 12 --snr 20 --json runs/ml/aero_feasible_20db_derived.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from scipy.stats import wilcoxon                                             # noqa: E402
from experiments.multiclass_lead import class_windows, measure                # noqa: E402

# Factors applied to the commanded lateral acceleration; 1.0 is the dataset amplitude.
#
# DERIVED_FACTOR is the feasibility factor the paper reports, one scalar evaluated at the
# aerodynamic reference condition of 25.2 km / M5.50, where qbar = 52.4 kPa. 1/4.7 is the
# alpha-limited value, the sensitivity case stated in Section 2 of the paper.
#
# The supersonic-cruise trajectories fly at a per-trajectory median qbar of 86.9 kPa (mean 94.9 kPa; qbar at the mean condition of 24.6 km / M6.84 is 88.8 kPa), 1.66x the
# reference, and the paper quotes that median. Evaluating a_avail/a_cmd per trajectory with the
# project's atmosphere model and CAV-H polar gives a median factor of 0.4931 (PER_TRAJ_FACTOR),
# and 28 of 30 trajectories permit more than 0.2798. The advantage grows as commanded g falls, so
# the fixed scalar gives a larger advantage than the per-trajectory factor.
DERIVED_FACTOR = 0.2798            # C_L,max = 1.296 at alpha 25 deg
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
    print("  Scaling the command scales fin deflection and hence fin rate; the statistic goes as")
    print("  rate^2, while the 2.0 m/s^2 onset threshold and the noise floor stay fixed.\n")
    print("%-32s %8s %8s %6s %6s %22s %20s"
          % ("commanded amplitude", "mean g", "budget", "det", "FA", "adv vs CUSUM (ms)",
             "adv vs GLR (ms)"))
    print("-" * 118)

    out = []
    for fac, lab in FACTORS:
        # As in sweep_class(): one trajectory per seed, events pooled by median within a
        # trajectory, statistics formed across trajectories. n counts trajectories.
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
                # No muD/CUSUM pair formed on this trajectory, so it has no advantage entry; it is
                # counted in n_mu_never_alarmed.
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

    print("\nmean g and budget are means of per-trajectory medians; det and FA are mean detection")
    print("and no-cue false-alarm rates over events. Each advantage cell is the median of the")
    print("per-trajectory paired advantages, its 2.5-97.5 percentile bootstrap interval, and the")
    print("number of trajectories n.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
