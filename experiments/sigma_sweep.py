"""experiments/sigma_sweep.py -- the advantage as a function of the comparator's noise.

WHY THIS IS THE HEADLINE EXPERIMENT AND NOT A ROBUSTNESS CHECK.

The comparator reads true lateral acceleration corrupted by Gaussian measurement noise of standard
deviation sigma. The micro-Doppler lead does not depend on sigma at all; only the comparator moves.
The reported advantage is therefore (muD lead - kinematic lead(sigma)), it is monotone in sigma, and
it changes sign near sigma ~ 0.005 m/s^2. At sigma = 0 the comparator wins.

What this script measures is a two-line result:

    the muD arm alarms within ~2 ms of the command and so converts essentially the whole airframe
    budget; a kinematic arm reading the true state at noise sigma converts a sigma-dependent
    fraction of it; here is the crossover.

The operating point must therefore be argued from a sensor model -- a radar cannot measure lateral
acceleration directly, it twice-differentiates noisy position -- and never assumed. Sec. II of the
letter carries that argument and this sweep is reported across sigma alongside it.

WHICH ARM IS THE COMPARATOR. Every kinematic arm in multiclass_lead.KIN_ARMS is measured here and
every one of them is written to the JSON under its own name, so the arm a number came from is
never ambiguous. The arms are genuinely different detectors: at sigma = 0.3 the trailing mean leads
by +13.2 ms and CUSUM by +29.2 ms, so a row mixing them would not close.

    REPORTED_ARM below names the comparator the letter quotes, `kin_lead_ms` is that arm's lead and
    nothing else, and `arm_lead_ms` / `arm_adv_median` carry every arm so the choice is auditable
    from the artifact alone.

Note that `adv_median` is the median of the PAIRED per-trajectory differences and is not required to
equal `muD_lead_ms - kin_lead_ms`, which is a difference of marginal medians over arms that survive
on different subsets -- the letter's own Section V says why that is not a pairing. The two agree to
about a millisecond here (17.0 vs 18.3); the paired figure is the one to quote.

    python experiments/sigma_sweep.py --seeds 30 --json runs/ml/sigma_sweep.json
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

# THE DEFAULT MUST BE THE AMPLITUDE THE LETTER REPORTS, so that the reproduction command printed
# in the manuscript -- "sigma_sweep.py --seeds 30" -- regenerates the table it claims to.
AMP = 0.2798                  # aero_feasible_factor.py: 1/3.57, the derived feasible amplitude
ALT_AMP = 1 / 4.7             # the alpha-limited alternative, quoted in Sec. II as a sensitivity

# The comparator the letter reports. Page CUSUM is a published detector in a published closed form;
# the trailing mean is ours and the GLR is degenerate at this dwell (identical alarm times), so
# neither may be quoted as the kinematic arm. Changing this string changes the reported comparator
# everywhere -- the JSON, the table and the figure all read it.
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
    print("  The muD lead should be CONSTANT down this table. If it is, the advantage is a")
    print("  statement about the comparator's assumed sensor quality, not about the channel.\n")
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
                        # the REPORTED comparator's lead, and nothing else
                        reported_arm=REPORTED_ARM,
                        kin_lead_ms=float(np.median(kins)) if kins else None,
                        adv_median=float(np.median(a)) if a.size else None,
                        adv_lo=lo, adv_hi=hi, p=float(p),
                        adv=[float(x) for x in a],
                        # every arm, so "which detector is this column?" is answerable from
                        # the artifact without reading this script
                        arm_lead_ms={nm: (float(np.median(lead[nm])) if lead[nm] else None)
                                     for nm in names},
                        arm_adv_median={nm: (float(np.median(advn[nm])) if advn[nm] else None)
                                        for nm in names},
                        arm_n={nm: len(advn[nm]) for nm in names},
                        # PER-TRAJECTORY leads, not just their medians. The paired advantage at
                        # the operating point is bimodal, and the split is in the comparator's
                        # alarm time rather than in the muD arm's -- a claim that cannot be
                        # checked, or refuted, from medians alone.
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
