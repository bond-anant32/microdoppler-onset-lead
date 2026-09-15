"""experiments/shape_period_sweep.py -- detection against the period of a sinusoidal command.

The paper's shape axis tests step, raised-cosine and sinusoid commands at dur = 0.30 s, plus one
raised-cosine at 1.00 s. The trajectory generator for the supersonic_cruise class commands

    a_lat_command(t) = weave_lat_accel * sin(2*pi*weave_cycles*m)

with a period of 12.4-19.1 s over the dataset's sampled weave_cycles and weave_dur, one to two
orders of magnitude slower than the published sweep. At that period the muD arm does not alarm
(true_shape_test.py, detection 0%). The statistic is the square of fin rate, and fin rate scales as
amplitude/period.

This script sweeps the sinusoid period over PERIODS at amplitude factor AMP and reports where
detection falls below 50%. The published raised-cosine at 1.0 s, with 10% detection, is consistent
with this curve.

    python experiments/shape_period_sweep.py --json runs/ml/shape_period.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AMP, SNR, REPS, DWELL, SIGMA, SEEDS = 0.2798, 40.0, 12, 0.002, 0.3, 30
PERIODS = (0.10, 0.30, 0.60, 1.00, 2.00, 4.00, 8.00, 15.00)


def run_cell(period):
    import numpy as np
    from scipy.stats import wilcoxon
    import experiments.multiclass_lead as ml
    from experiments.multiclass_lead import class_windows, measure

    ml.SHAPES = dict(ml.SHAPES)
    ml.SHAPES["supersonic_cruise"] = ("sinusoid", float(period))

    advs, mus, kins, dets = [], [], [], []
    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        if not wins:
            continue
        ea, em, ek = [], [], []
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
            m = measure(w2, SNR, REPS, DWELL, kin_noise=SIGMA)
            if not m:
                continue
            dets.append(m["det"])
            if m.get("muD") is not None:
                em.append(m["muD"])
            r = (m.get("arms") or {}).get("CUSUM Page54") or {}
            if r.get("lead") is not None:
                ek.append(r["lead"])
            if r.get("adv") is not None:
                ea.append(r["adv"])
        if ea:
            advs.append(float(np.median(ea)))
        if em:
            mus.append(float(np.median(em)))
        if ek:
            kins.append(float(np.median(ek)))

    a = np.asarray(advs, float)
    med = lambda v: float(np.median(v)) if len(v) else None                   # noqa: E731
    return dict(period_s=float(period), n=int(a.size), adv_median=med(a),
                worst=float(a.min()) if a.size else None, n_pos=int((a > 0).sum()),
                muD_lead_ms=med(mus), cusum_lead_ms=med(kins),
                det=float(np.mean(dets)) if dets else None,
                p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--procs", type=int, default=max(1, min(8, cpu_count() - 2)))
    args = ap.parse_args()

    print("Command-period sweep: the statistic scales as (fin rate)^2 and falls with period")
    print("  the generator's weave period for this class is 12.4-19.1 s\n")
    with Pool(args.procs, maxtasksperchild=1) as pool:
        rows = pool.map(run_cell, PERIODS, chunksize=1)

    print("%10s %5s %11s %9s %6s %10s %10s %7s"
          % ("period s", "n", "advantage", "worst", "pos", "muD lead", "CUSUM", "det"))
    print("-" * 78)
    for r in rows:
        f = lambda v, p="%+.2f": (p % v) if v is not None else "--"           # noqa: E731
        print("%10.2f %5d %11s %9s %3d/%-2d %10s %10s %6.0f%%"
              % (r["period_s"], r["n"], f(r["adv_median"]), f(r["worst"], "%+.1f"),
                 r["n_pos"], r["n"], f(r["muD_lead_ms"]), f(r["cusum_lead_ms"]),
                 100 * (r["det"] or 0)))

    live = [r for r in rows if (r["det"] or 0) >= 0.5]
    dead = [r for r in rows if (r["det"] or 0) < 0.5]
    if live and dead:
        print("\nDetection >=50%% up to a command period of %.2f s; <50%% from %.2f s."
              % (max(r["period_s"] for r in live), min(r["period_s"] for r in dead)))
    print("For comparison, the generator commands a weave period of 12.4-19.1 s.")
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(sigma=SIGMA, snr_db=SNR, seeds=SEEDS, reps=REPS, amp_factor=AMP,
                       periods=list(PERIODS), rows=rows), open(args.json, "w"),
                  indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
