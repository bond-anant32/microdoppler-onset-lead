"""experiments/slow_cue_detector.py -- muD detector variants at long command periods.

shape_period_sweep.py shows the base muD arm's detection rate falling with command period: 97% at
0.1 s, 43% at 1.0 s and zero from 4.0 s. The generator commands the supersonic_cruise weave at
periods of 12.4-19.1 s. This script separates a limitation of the cue from a limitation of the
detector. The base muD statistic is

    z(t) = trailing mean of |dphi/dt|^2 over D = 2 ms,   alarm when z exceeds the cruise maximum

which is matched to a millisecond burst. The kinematic arm it races uses a CUSUM. Two properties
of the base statistic penalise slow commands:

  * z ~ (fin rate)^2 ~ (A/T)^2, so the signal falls as 1/T^2 while the noise floor of a 2 ms
    trailing mean stays fixed;
  * a max-threshold ignores persistence, and a slow cue is a small sustained excursion, the
    alternative a CUSUM is designed to detect.

All variants share the cue, the causal rule, the zero-false-alarm threshold construction and the
3-of-3 alarm, and differ only in the muD statistic:

    base      D = 2 ms,  max-threshold
    dwellNN   D = NN ms, max-threshold          (dwell matched to a slower event)
    cusum     D = 2 ms,  CUSUM on z             (accumulates persistence, as the comparator does)
    cusumNN   D = NN ms, CUSUM on z             (both)

The kinematic arm is the same for every variant, and each variant's false-alarm rate is measured
on a no-cue null. A variant with detection >= 50% at null FA < 5% at the longest period places the
long-period loss in the detector. If no variant reaches that bar, the loss belongs to the cue.

    python experiments/slow_cue_detector.py --json runs/ml/slow_cue.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AMP, SNR, REPS, SIGMA, SEEDS = 0.2798, 40.0, 12, 0.3, 30
BASE_DWELL = 0.002
PERIODS = (1.0, 4.0, 15.0)
# (tag, dwell_s, use_cusum)
VARIANTS = (
    ("base  D=2ms max", 0.002, False),
    ("D=20ms max", 0.020, False),
    ("D=100ms max", 0.100, False),
    ("cusum D=2ms", 0.002, True),
    ("cusum D=20ms", 0.020, True),
    ("cusum D=100ms", 0.100, True),
)


def run_cell(arg):
    period, tag, dwell, use_cusum = arg
    import numpy as np
    from scipy.stats import wilcoxon
    import experiments.multiclass_lead as ml
    from experiments.multiclass_lead import class_windows
    from experiments.dphi_sweep import return_from_fin, stat_matched_phase
    from experiments.causal_dwell_test import causal_lead, thr_from_cruise, DT_R

    ml.SHAPES = dict(ml.SHAPES)
    ml.SHAPES["supersonic_cruise"] = ("sinusoid", float(period))

    advs, mus, kins, dets, fas = [], [], [], [], []
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
            fl = ml.drive_airframe(w2["t"], w2["a_cmd"], w2["V"], w2["alt"])
            t_on, _ = ml.onset_from_achieved(fl["t"], fl["az"])
            if t_on is None:
                continue
            tf, delta, az = fl["t"], fl["delta"], fl["az"]
            for r in range(REPS):
                # ---- muD arm: the variant under test -------------------------------------------
                t, s = return_from_fin(tf, delta, SNR, 4000 + r, ml.FIN_ARM_M)
                z = stat_matched_phase(t, s, dwell)
                if use_cusum:
                    z = ml.stat_cusum(np.asarray(z, float), dwell)
                lm = causal_lead(t, z, thr_from_cruise(t, z, t_on), t_on)
                # ---- kinematic arm: identical for every variant --------------------------------
                rk = np.random.default_rng(4000 + r + 991)
                azr = np.abs(np.interp(t, tf, az) + rk.normal(0, SIGMA, len(t)))
                zk = ml.stat_cusum(azr, BASE_DWELL)
                lk = causal_lead(t, zk, thr_from_cruise(t, zk, t_on), t_on)
                dets.append(1.0 if lm is not None else 0.0)
                # ---- no-cue null for the same variant ------------------------------------------
                tn, sn = return_from_fin(tf, delta, SNR, 4000 + r, ml.FIN_ARM_M, cue_on=False)
                zn = stat_matched_phase(tn, sn, dwell)
                if use_cusum:
                    zn = ml.stat_cusum(np.asarray(zn, float), dwell)
                fas.append(1.0 if causal_lead(tn, zn, thr_from_cruise(tn, zn, t_on), t_on)
                           is not None else 0.0)
                if lm is not None:
                    em.append(1000 * lm)
                if lk is not None:
                    ek.append(1000 * lk)
                if lm is not None and lk is not None:
                    ea.append(1000 * (lm - lk))
        if ea:
            advs.append(float(np.median(ea)))
        if em:
            mus.append(float(np.median(em)))
        if ek:
            kins.append(float(np.median(ek)))

    a = np.asarray(advs, float)
    med = lambda v: float(np.median(v)) if len(v) else None                   # noqa: E731
    return dict(period_s=float(period), variant=tag, dwell_s=dwell, cusum=bool(use_cusum),
                n=int(a.size), adv_median=med(a), worst=float(a.min()) if a.size else None,
                n_pos=int((a > 0).sum()), muD_lead_ms=med(mus), cusum_lead_ms=med(kins),
                det=float(np.mean(dets)) if dets else None,
                fa=float(np.mean(fas)) if fas else None,
                p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--procs", type=int, default=max(1, min(9, cpu_count() - 2)))
    args = ap.parse_args()

    print("Slow-cue detector variants at long command periods")
    print("  kinematic arm fixed; only the muD statistic varies. FA is the no-cue null.\n")
    jobs = [(p, t, d, c) for p in PERIODS for (t, d, c) in VARIANTS]
    with Pool(args.procs, maxtasksperchild=1) as pool:
        rows = pool.map(run_cell, jobs, chunksize=1)

    for p in PERIODS:
        print("--- command period %.1f s %s" % (p, "-" * 48))
        print("  %-18s %5s %11s %9s %6s %10s %7s %6s"
              % ("variant", "n", "advantage", "worst", "pos", "muD lead", "det", "FA"))
        for r in [x for x in rows if x["period_s"] == p]:
            f = lambda v, q="%+.2f": (q % v) if v is not None else "--"       # noqa: E731
            print("  %-18s %5d %11s %9s %3d/%-2d %10s %6.0f%% %5.0f%%"
                  % (r["variant"], r["n"], f(r["adv_median"]), f(r["worst"], "%+.1f"),
                     r["n_pos"], r["n"], f(r["muD_lead_ms"]), 100 * (r["det"] or 0),
                     100 * (r["fa"] or 0)))
        print()

    slow = [r for r in rows if r["period_s"] == max(PERIODS)]
    best = max(slow, key=lambda r: (r["det"] or 0.0))
    print("At the generator's own timescale (%.0f s) the best variant is %r: detection %.0f%%, "
          "null FA %.0f%%." % (max(PERIODS), best["variant"], 100 * (best["det"] or 0),
                               100 * (best["fa"] or 0)))
    if (best["det"] or 0) >= 0.5 and (best["fa"] or 1) < 0.05:
        print("=> A variant recovers detection at the null false-alarm bar, so the long-period")
        print("   loss is a property of the detector statistic.")
    else:
        print("=> No variant recovers detection at the null false-alarm bar, so the long-period")
        print("   loss is a property of the cue.")
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(snr_db=SNR, sigma=SIGMA, seeds=SEEDS, reps=REPS, amp_factor=AMP,
                       periods=list(PERIODS), rows=rows), open(args.json, "w"),
                  indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
