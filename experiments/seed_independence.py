"""experiments/seed_independence.py -- how much does the operating point owe to one noise block?

WHAT IS BEING TESTED. measure() in experiments/multiclass_lead.py seeds the micro-Doppler arm at
4000 + r and the comparator at 4991 + r, where r is the REPETITION index. The trajectory index
never enters either seed, so all 30 trajectories are measured on the same set of noise
realizations. That is common random numbers: it is a legitimate variance-reduction choice and it
tightens the paired comparison. It also means the 30 per-trajectory values entering the bootstrap
and the signed-rank test share a noise block, so those intervals are conditional on the block and
do not carry the variation BETWEEN blocks.

This measures that variation directly. Each cell re-runs the operating point with a per-trajectory
offset added to BOTH seeds:

  * within a trajectory the two arms stay on the same realization index, so the pairing that
    licenses the signed-rank test is preserved exactly;
  * across trajectories the streams are disjoint, so the shared-block structure is removed.

Offset 0 is the CONTROL and must reproduce runs/ml/sigma_sweep_derived.json's sigma = 0.3 row. If
it does not, the harness is not measuring the shipped quantity and no other row means anything.

WHY THE SEEDS ARE PATCHED RATHER THAN THREADED. measure() takes no seed argument. Adding one would
edit the shared engine and invalidate every artifact's freshness for a change that is provably
behaviour-preserving, so the two seed windows measure() actually uses are remapped here instead.
Nothing else is touched: class_windows keeps 90000 + sd, so the trajectories are identical to the
shipped run, and the bootstrap keeps its own seed.

    python experiments/seed_independence.py --json runs/ml/seed_independence.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from scipy.stats import wilcoxon                                             # noqa: E402
from experiments.multiclass_lead import class_windows, measure                # noqa: E402
from experiments.sigma_sweep import AMP, REPORTED_ARM, boot                   # noqa: E402

REPS, SNR, DWELL, SEEDS, SIGMA = 12, 40.0, 0.002, 30, 0.3
OFFSETS = (0, 100000, 200000, 300000, 500000, 700000, 1100000)

_real_rng = np.random.default_rng
_OFF = {"v": 0}
_MU = range(4000, 4000 + REPS)
_KIN = range(4991, 4991 + REPS)


def _patched(seed=None, *a, **k):
    """Remap ONLY the two seed windows measure() draws its noise from."""
    if isinstance(seed, (int, np.integer)):
        s = int(seed)
        if s in _MU or s in _KIN:
            seed = s + _OFF["v"]
    return _real_rng(seed, *a, **k)


np.random.default_rng = _patched


def cell(offset_per_traj):
    adv = []
    for sd in range(SEEDS):
        _OFF["v"] = offset_per_traj * (sd + 1)
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=_real_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        if not wins:
            continue
        ev = []
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
            m = measure(w2, SNR, REPS, DWELL, kin_noise=SIGMA)
            if not m:
                continue
            r = (m.get("arms") or {}).get(REPORTED_ARM) or {}
            if r.get("adv") is not None:
                ev.append(r["adv"])
        if ev:
            adv.append(float(np.median(ev)))
    _OFF["v"] = 0
    a = np.asarray(adv, float)
    if a.size < 3:
        return dict(offset=int(offset_per_traj), n=int(a.size),
                    adv_median=None, adv_lo=None, adv_hi=None, p=None)
    lo, hi = boot(a)
    return dict(offset=int(offset_per_traj), n=int(a.size),
                adv_median=float(np.median(a)), adv_lo=float(lo), adv_hi=float(hi),
                p=float(wilcoxon(a).pvalue) if np.any(a != 0) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("SEED INDEPENDENCE AT THE OPERATING POINT (sigma = %.1f m/s^2)\n" % SIGMA)
    print("  offset 0 is the control and must reproduce the shipped sigma=0.3 row.\n")
    print("%-12s %5s %10s %20s %10s" % ("offset", "n", "median", "bootstrap 95%", "p"))
    print("-" * 62)
    rows = []
    for off in OFFSETS:
        r = cell(off)
        rows.append(r)
        print("%-12s %5d %10s %20s %10s"
              % ("shipped" if off == 0 else "+%d" % off, r["n"],
                 "%+.2f" % r["adv_median"] if r["adv_median"] is not None else "--",
                 "[%+.2f,%+.2f]" % (r["adv_lo"], r["adv_hi"])
                 if r["adv_lo"] is not None else "--",
                 "%.1e" % r["p"] if r["p"] is not None else "--"))

    ind = [r["adv_median"] for r in rows
           if r["offset"] and r["adv_median"] is not None]
    out = dict(sigma=SIGMA, snr_db=SNR, reps=REPS, dwell_s=DWELL, seeds=SEEDS,
               amp_factor=AMP, arm=REPORTED_ARM, offsets=list(OFFSETS), rows=rows,
               shipped=rows[0]["adv_median"],
               independent_n=len(ind),
               independent_median=float(np.median(ind)) if ind else None,
               independent_lo=min(ind) if ind else None,
               independent_hi=max(ind) if ind else None,
               all_positive=bool(ind) and all(v > 0 for v in ind))
    print("\n  shipped %+.2f ms | %d independent blocks: median %+.2f, range %+.2f to %+.2f, "
          "all positive %s"
          % (out["shipped"], out["independent_n"], out["independent_median"],
             out["independent_lo"], out["independent_hi"], out["all_positive"]))
    print("\n  The shipped bootstrap interval is formed over trajectories WITHIN one block and")
    print("  does not carry the between-block spread above.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
