"""experiments/joint_mc_sweep.py -- JOINT randomisation of every free axis at once.

WHY. sensitivity_sweep.py varies one constant at a time (OAT) and reports that the sign is positive
in all 62 cells where every trajectory pairs. Three independent audits made the same correct
objection:

    OAT is a local method. It cannot detect interactions, and the axes here are NOT independent --
    CN_delta sets both how much direct fin lift the kinematic arm sees AND part of the cue; servo
    bandwidth sets both delta(t) and a_z(t). "Positive in all 62 cells" is a statement about 62
    points on the axes, not about the parameter VOLUME. There may be a corner -- low CN_delta x
    diluted RCS x low SNR x non-step command -- where the advantage vanishes or reverses, and OAT
    is structurally blind to it.

That objection is right and this script answers it: draw ALL the free axes simultaneously and
independently on every trial, so interaction corners are sampled rather than excluded by
construction.

AXES DRAWN JOINTLY (ranges are the OAT sweep's own endpoints, so this covers the same volume):
    sigma        log-uniform 0.03 .. 3.0 m/s^2   (above the measured crossover; below it the letter
                                                  already reports the comparator winning)
    snr_db       uniform 20 .. 50 dB
    CN_delta     uniform 0.05 .. 1.20
    wn_servo     log-uniform 25 .. 300 rad/s     (5 and 10 are excluded: the OAT run shows they do
                                                  not pair, n=4 and n=19 of 30, so a draw there
                                                  reports a selected subset rather than a property)
    rate_max_dps uniform 50 .. 800 deg/s
    amp_factor   uniform 0.2128 .. 0.4931        (the three defensible feasibility readings)
    rho          uniform 0.05 .. 1.0             (fin's share of return power -- the dilution axis)
    shape        {step, raised-cosine, sinusoid}  (the authored-command axis)

REPORTED: the fraction of trials with a positive median advantage, the distribution of the
magnitude, and -- the point of the exercise -- the trials where the sign REVERSES, with the corner
that produced them, so an interaction can be named rather than assumed absent.

PAIRING DISCIPLINE IS UNCHANGED. A trial is only counted if all 30 trajectories pair; partial trials
are reported separately, exactly as the OAT sweep does, because a statistic from a partially paired
cell is filtered by a criterion correlated with the outcome.

    python experiments/joint_mc_sweep.py --trials 60 --json runs/ml/joint_mc_sweep.json
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPS, SEEDS = 12, 30
SHAPES = ("step", "raised-cosine", "sinusoid")


def draw(rng):
    """One point in the joint parameter volume."""
    return dict(
        sigma=float(10 ** rng.uniform(math.log10(0.03), math.log10(3.0))),
        snr_db=float(rng.uniform(20.0, 50.0)),
        CN_delta=float(rng.uniform(0.05, 1.20)),
        wn_servo=float(10 ** rng.uniform(math.log10(25.0), math.log10(300.0))),
        rate_max_dps=float(rng.uniform(50.0, 800.0)),
        amp_factor=float(rng.uniform(0.2128, 0.4931)),
        rho=float(rng.uniform(0.05, 1.0)),
        shape=str(SHAPES[int(rng.integers(0, len(SHAPES)))]),
    )


def run_trial(arg):
    idx, p = arg
    import numpy as np
    import experiments.multiclass_lead as ml
    import experiments.dphi_sweep as dps
    from experiments.multiclass_lead import class_windows, measure
    from experiments.causal_dwell_test import DT_R, C, FC_HZ

    # ---- plant: patch the binding drive_airframe actually reads, copy so workers cannot share ----
    af = dict(ml.DEFAULT_AIRFRAME)
    af["CN_delta"] = p["CN_delta"]
    af["wn_s"] = p["wn_servo"]
    af["rate_max"] = math.radians(p["rate_max_dps"])
    ml.DEFAULT_AIRFRAME = af

    # ---- command shape: the authored axis -------------------------------------------------------
    ml.SHAPES = dict(ml.SHAPES)
    dur = {"step": 0.0, "raised-cosine": 30.0, "sinusoid": 36.0}[p["shape"]]
    ml.SHAPES["supersonic_cruise"] = (p["shape"], dur)

    # ---- signal model: dilution with a RANDOM-phase interferer ----------------------------------
    rho = p["rho"]

    def return_multi(t_fin, delta, snr_db, seed, fin_arm, shift_s=0.0, cue_on=True):
        rng = np.random.default_rng(seed)
        t = np.arange(t_fin[0], t_fin[-1], DT_R)
        d = (np.interp(t - shift_s, t_fin, delta, left=delta[0], right=delta[-1])
             if cue_on else np.zeros_like(t))
        lam = C / FC_HZ
        phase = 4.0 * np.pi * (fin_arm * np.sin(d)) / lam
        g = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
        s = np.sqrt(rho) * np.exp(1j * phase) + np.sqrt(1.0 - rho) * g
        p_n = 1.0 / (10.0 ** (snr_db / 10.0))
        return t, s + np.sqrt(p_n / 2.0) * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))

    dps.return_from_fin = return_multi
    ml.return_from_fin = return_multi

    advs, dets = [], []
    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd),
                                    amp_factor=p["amp_factor"])
        except Exception:                                                     # noqa: BLE001
            continue
        if not wins:
            continue
        ev = []
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * p["amp_factor"]
            try:
                m = measure(w2, p["snr_db"], REPS, 0.002, kin_noise=p["sigma"])
            except Exception:                                                 # noqa: BLE001
                continue
            if not m:
                continue
            dets.append(m["det"])
            r = (m.get("arms") or {}).get("CUSUM Page54") or {}
            if r.get("adv") is not None:
                ev.append(r["adv"])
        if ev:
            advs.append(float(np.median(ev)))

    a = np.asarray(advs, float)
    return dict(trial=idx, **p, n=int(a.size),
                adv_median=float(np.median(a)) if a.size else None,
                worst=float(a.min()) if a.size else None,
                n_pos=int((a > 0).sum()),
                det=float(np.mean(dets)) if dets else None,
                full_pairing=bool(a.size == SEEDS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--procs", type=int, default=max(1, min(12, cpu_count() - 2)))
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(args.seed)
    pts = [(i, draw(rng)) for i in range(args.trials)]

    print("JOINT MONTE CARLO -- every free axis drawn simultaneously")
    print("  %d trials x %d trajectories x %d reps. OAT sweeps cannot see interactions; this can.\n"
          % (args.trials, SEEDS, REPS))
    with Pool(args.procs, maxtasksperchild=1) as pool:
        rows = pool.map(run_trial, pts, chunksize=1)

    full = [r for r in rows if r["full_pairing"] and r["adv_median"] is not None]
    part = [r for r in rows if not r["full_pairing"]]
    neg = [r for r in full if r["adv_median"] <= 0]

    print("%5s %7s %6s %8s %8s %8s %7s %6s %-14s %4s %9s"
          % ("trial", "sigma", "SNR", "CN_del", "wn_srv", "rate", "amp", "rho", "shape", "n", "adv"))
    print("-" * 104)
    for r in sorted(rows, key=lambda r: (r["adv_median"] is None, r["adv_median"] or 0)):
        print("%5d %7.3f %6.1f %8.2f %8.1f %8.0f %7.3f %6.2f %-14s %4d %9s"
              % (r["trial"], r["sigma"], r["snr_db"], r["CN_delta"], r["wn_servo"],
                 r["rate_max_dps"], r["amp_factor"], r["rho"], r["shape"], r["n"],
                 ("%+.2f" % r["adv_median"]) if r["adv_median"] is not None else "--"))

    print("\nfull-pairing trials : %d of %d" % (len(full), len(rows)))
    if full:
        v = np.array([r["adv_median"] for r in full])
        print("  positive           : %d of %d (%.0f%%)"
              % ((v > 0).sum(), v.size, 100.0 * (v > 0).sum() / v.size))
        print("  magnitude          : min %+.2f  median %+.2f  max %+.2f ms"
              % (v.min(), np.median(v), v.max()))
    print("partial-pairing     : %d (reported, not pooled -- selected subsets)" % len(part))
    if neg:
        print("\n*** SIGN REVERSALS IN THE JOINT VOLUME (%d) -- the corner OAT could not see:" % len(neg))
        for r in neg:
            print("    trial %d: adv %+.2f at sigma=%.3f SNR=%.1f CN_delta=%.2f wn_servo=%.0f "
                  "rho=%.2f shape=%s"
                  % (r["trial"], r["adv_median"], r["sigma"], r["snr_db"], r["CN_delta"],
                     r["wn_servo"], r["rho"], r["shape"]))
    else:
        print("\nNo sign reversal among full-pairing trials in the joint volume.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(trials=args.trials, seed=args.seed, seeds=SEEDS, reps=REPS,
                       n_full=len(full), n_partial=len(part), n_negative=len(neg),
                       rows=rows), open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
