"""experiments/sigma_sweep.py -- lead advantage as a function of the comparator's noise sigma.

The comparator reads true lateral acceleration corrupted by Gaussian measurement noise of standard
deviation sigma (m/s^2). The micro-Doppler lead does not depend on sigma, so the advantage
(muD lead - kinematic lead(sigma)) varies through the comparator alone. It is monotone in sigma and
changes sign between sigma = 0, where the comparator leads by 1 ms, and sigma = 0.003 m/s^2.

The muD arm alarms within ~2 ms of the command and so uses almost the whole airframe budget. A
kinematic arm reading the true state at noise sigma uses a sigma-dependent fraction of it. A radar
obtains lateral acceleration by differentiating noisy position twice, so the value of sigma at the
operating point comes from a sensor model, given in Section 4 of the paper.

Every kinematic arm in multiclass_lead.KIN_ARMS is measured and written to the JSON under its own
name. At sigma = 0.3 the trailing mean leads by +13.2 ms and CUSUM by +29.2 ms. REPORTED_ARM names
the comparator the paper quotes; `kin_lead_ms` is that arm's lead, and `arm_lead_ms` and
`arm_adv_median` hold every arm.

`adv_median` is the median of the paired per-trajectory differences. It need not equal
`muD_lead_ms - kin_lead_ms`, a difference of marginal medians over arms that survive on different
subsets (Section V of the paper). At sigma = 0.3 the two are 17.0 and 18.3 ms; the paper reports
the paired figure.

    python experiments/sigma_sweep.py --seeds 30 --amp-factor 0.2798 --json runs/ml/sigma_sweep_derived.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from scipy.stats import wilcoxon                                             # noqa: E402
from experiments.multiclass_lead import class_windows, measure, KIN_ARMS      # noqa: E402

SIGMAS = (0.0, 0.003, 0.005, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

# The default amplitude factor is the one the paper reports, so "sigma_sweep.py --seeds 30"
# regenerates Table 2.
AMP = 0.2798                  # derived CAV-H feasibility factor, 1/3.57
ALT_AMP = 1 / 4.7             # alpha-limited factor, the sensitivity case in Section 2 of the paper

# Comparator reported in the paper: Page's CUSUM (Page 1954), a published detector with a closed
# form. The trailing-mean arm is defined in this code, and at this dwell the GLR arm gives the same
# alarm times as the trailing mean. The JSON, the printed table and the figure all read this name.
REPORTED_ARM = "CUSUM Page54"


def boot(v, seed=11, n=20000):
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
    ap.add_argument("--snr", type=float, default=40.0)
    ap.add_argument("--dwell", type=float, default=0.002)
    ap.add_argument("--amp-factor", type=float, default=AMP)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("COMPARATOR-NOISE SWEEP -- supersonic_cruise, amplitude x%.3f, %.0f dB, %d seeds\n"
          % (args.amp_factor, args.snr, args.seeds))
    print("  sigma enters the comparator only, so the muD lead is expected to be constant down")
    print("  this table and the advantage to follow the comparator's sensor noise.\n")
    print("  Reported comparator: %s. Every arm is measured and written to the JSON.\n"
          % REPORTED_ARM)
    print("%8s %10s %10s %6s %26s %8s"
          % ("sigma", "muD lead", "kin lead", "n", "advantage vs %s (ms)" % REPORTED_ARM, "p"))
    print("-" * 78)

    names = [n for n, _ in KIN_ARMS]
    out = []
    for sig in SIGMAS:
        mus = []
        lead = {n: [] for n in names}
        advn = {n: [] for n in names}
        n_entered = n_mu_never = 0
        for sd in range(args.seeds):
            try:
                wins, _ = class_windows("supersonic_cruise",
                                        rng=np.random.default_rng(90000 + sd),
                                        amp_factor=args.amp_factor)
            except Exception:                                                 # noqa: BLE001
                continue
            if not wins:
                continue
            n_entered += 1
            ev_mu = []
            ev_lead = {n: [] for n in names}
            ev_adv = {n: [] for n in names}
            for w in wins:
                w2 = dict(w)
                w2["a_cmd"] = np.asarray(w["a_cmd"], float) * args.amp_factor
                m = measure(w2, args.snr, args.reps, args.dwell, kin_noise=sig)
                if not m:
                    continue
                if m.get("muD") is not None:
                    ev_mu.append(m["muD"])
                arms = m.get("arms") or {}
                for nm in names:
                    r = arms.get(nm) or {}
                    if r.get("lead") is not None:
                        ev_lead[nm].append(r["lead"])
                    if r.get("adv") is not None:
                        ev_adv[nm].append(r["adv"])
            if ev_mu:
                mus.append(float(np.median(ev_mu)))
            for nm in names:
                if ev_lead[nm]:
                    lead[nm].append(float(np.median(ev_lead[nm])))
                if ev_adv[nm]:
                    advn[nm].append(float(np.median(ev_adv[nm])))
            if not ev_adv[REPORTED_ARM] and (ev_mu or any(ev_lead.values())):
                n_mu_never += 1

        kins = lead[REPORTED_ARM]
        a = np.asarray(advn[REPORTED_ARM], float)
        if a.size >= 3:
            lo, hi = boot(a)
            p = wilcoxon(a).pvalue if np.any(a != 0) else float("nan")
            cell = "%+.1f [%+.1f,%+.1f]" % (np.median(a), lo, hi)
        else:
            lo = hi = p = float("nan")
            cell = "n too small"
        print("%8.3f %10s %10s %6d %26s %8.1e"
              % (sig,
                 ("%+.1f" % np.median(mus)) if mus else "--",
                 ("%+.1f" % np.median(kins)) if kins else "--",
                 a.size, cell, p))
        out.append(dict(sigma=sig, n=int(a.size), n_entered=n_entered,
                        n_mu_never_alarmed=n_mu_never,
                        muD_lead_ms=float(np.median(mus)) if mus else None,
                        # lead of the reported comparator
                        reported_arm=REPORTED_ARM,
                        kin_lead_ms=float(np.median(kins)) if kins else None,
                        adv_median=float(np.median(a)) if a.size else None,
                        adv_lo=lo, adv_hi=hi, p=float(p),
                        adv=[float(x) for x in a],
                        # leads, advantages and pair counts for every arm, keyed by arm name
                        arm_lead_ms={nm: (float(np.median(lead[nm])) if lead[nm] else None)
                                     for nm in names},
                        arm_adv_median={nm: (float(np.median(advn[nm])) if advn[nm] else None)
                                        for nm in names},
                        arm_n={nm: len(advn[nm]) for nm in names},
                        # Per-trajectory leads. The paired advantage at the operating point is
                        # bimodal, with the split in the comparator's alarm time, and these
                        # lists show it.
                        muD_lead_list=[float(x) for x in mus],
                        arm_lead_list={nm: [float(x) for x in lead[nm]] for nm in names},
                        snr_db=args.snr, reps=args.reps, dwell_s=args.dwell,
                        amp_factor=args.amp_factor, seeds=args.seeds))

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
