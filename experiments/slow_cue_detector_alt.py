"""experiments/slow_cue_detector_alt.py -- Task 1: testing alternative detector families on slow commands (T=15s)
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import wilcoxon

AMP, SNR, REPS, SIGMA, SEEDS = 0.2798, 40.0, 12, 0.3, 30
BASE_DWELL = 0.002
PERIODS = (1.0, 4.0, 15.0)

# Statistics functions for alternative detector families:

def stat_band_power(t, s, dwell_s, period_s):
    """Family 1: Band power at the weave frequency (1/T)."""
    import experiments.dphi_sweep as ds
    f0 = 1.0 / period_s
    lo = max(0.001, f0 * 0.8)
    hi = f0 * 1.2
    res = ds.stat_band_phase(t, s, dwell_s, lo=lo, hi=hi)
    if res is None:
        return "unformable"
    return res

def stat_matched_filter(t, s, dwell_s, period_s):
    """Family 2: Matched filter to the known command shape derivative (cos(2pi t / T))."""
    import experiments.dphi_sweep as ds
    dphi_val = ds.dphi(s)
    dt_r = ds.DT_R
    n = max(2, int(dwell_s / dt_r))
    t_temp = np.arange(n) * dt_r
    omega = 2.0 * np.pi / period_s
    template = np.abs(np.cos(omega * t_temp))
    template = template / (np.linalg.norm(template) + 1e-12)
    out = np.convolve(dphi_val, template[::-1], mode="full")[:len(t)]
    return out

def stat_phase_deviation(t, s, dwell_s):
    """Family 3: Unwrapped phase deviation from cruise mean (envelope/magnitude rather than rate)."""
    import experiments.dphi_sweep as ds
    ph = np.unwrap(np.angle(s))
    dt_r = ds.DT_R
    n = max(2, int(dwell_s / dt_r))
    # cruise baseline from the first n samples
    ph_baseline = np.mean(ph[:n])
    dev = np.abs(ph - ph_baseline)
    out = np.convolve(dev, np.ones(n) / n, mode="full")[:len(t)]
    return out


def run_cell(arg):
    period, fam_name, dwell_s, use_cusum = arg
    import experiments.multiclass_lead as ml
    from experiments.multiclass_lead import class_windows
    from experiments.dphi_sweep import return_from_fin, stat_matched_phase
    from experiments.causal_dwell_test import causal_lead, thr_from_cruise

    ml.SHAPES = dict(ml.SHAPES)
    ml.SHAPES["supersonic_cruise"] = ("sinusoid", float(period))

    # Calculate statistic
    advs, mus, kins, dets, fas = [], [], [], [], []
    unformable = False

    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:
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
                # muD arm
                t, s = return_from_fin(tf, delta, SNR, 4000 + r, ml.FIN_ARM_M)
                if fam_name == "band_power":
                    z = stat_band_power(t, s, dwell_s, period)
                    if isinstance(z, str) and z == "unformable":
                        unformable = True
                        break
                elif fam_name == "matched_filter":
                    z = stat_matched_filter(t, s, dwell_s, period)
                elif fam_name == "phase_deviation":
                    z = stat_phase_deviation(t, s, dwell_s)
                else:
                    z = stat_matched_phase(t, s, dwell_s)

                if use_cusum:
                    z = ml.stat_cusum(np.asarray(z, float), dwell_s)
                lm = causal_lead(t, z, thr_from_cruise(t, z, t_on), t_on)

                # kinematic arm
                rk = np.random.default_rng(4000 + r + 991)
                azr = np.abs(np.interp(t, tf, az) + rk.normal(0, SIGMA, len(t)))
                zk = ml.stat_cusum(azr, BASE_DWELL)
                lk = causal_lead(t, zk, thr_from_cruise(t, zk, t_on), t_on)
                dets.append(1.0 if lm is not None else 0.0)

                # no-cue null
                tn, sn = return_from_fin(tf, delta, SNR, 4000 + r, ml.FIN_ARM_M, cue_on=False)
                if fam_name == "band_power":
                    zn = stat_band_power(tn, sn, dwell_s, period)
                elif fam_name == "matched_filter":
                    zn = stat_matched_filter(tn, sn, dwell_s, period)
                elif fam_name == "phase_deviation":
                    zn = stat_phase_deviation(tn, sn, dwell_s)
                else:
                    zn = stat_matched_phase(tn, sn, dwell_s)

                if use_cusum:
                    zn = ml.stat_cusum(np.asarray(zn, float), dwell_s)
                fas.append(1.0 if causal_lead(tn, zn, thr_from_cruise(tn, zn, t_on), t_on)
                           is not None else 0.0)

                if lm is not None:
                    em.append(1000 * lm)
                if lk is not None:
                    ek.append(1000 * lk)
                if lm is not None and lk is not None:
                    ea.append(1000 * (lm - lk))
            if unformable:
                break

        if ea:
            advs.append(float(np.median(ea)))
        if em:
            mus.append(float(np.median(em)))
        if ek:
            kins.append(float(np.median(ek)))

    if unformable:
        return dict(period_s=float(period), variant=f"{fam_name} D={dwell_s}s {'cusum' if use_cusum else 'max'}",
                    unformable=True, n=0, det=0.0, fa=0.0)

    a = np.asarray(advs, float)
    med = lambda v: float(np.median(v)) if len(v) else None
    return dict(period_s=float(period), variant=f"{fam_name} D={dwell_s}s {'cusum' if use_cusum else 'max'}",
                unformable=False,
                n=int(a.size), adv_median=med(a), worst=float(a.min()) if a.size else None,
                n_pos=int((a > 0).sum()), muD_lead_ms=med(mus), cusum_lead_ms=med(kins),
                det=float(np.mean(dets)) if dets else 0.0,
                fa=float(np.mean(fas)) if fas else 0.0,
                p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="runs/ml/slow_cue_alt.json")
    ap.add_argument("--procs", type=int, default=max(1, min(8, cpu_count() - 2)))
    args = ap.parse_args()

    print("SLOW-CUE DETECTOR ALT -- testing alternative families on slow command (T=15s)")

    variants = [
        # (fam_name, dwell_s, use_cusum)
        ("band_power", 0.002, False),
        ("band_power", 0.020, False),
        ("band_power", 0.100, False),
        ("band_power", 1.000, False),
        ("band_power", 15.000, False),
        ("band_power", 15.000, True),
        ("matched_filter", 0.002, False),
        ("matched_filter", 0.020, False),
        ("matched_filter", 0.100, False),
        ("matched_filter", 1.000, False),
        ("matched_filter", 1.000, True),
        ("matched_filter", 15.000, True),
        ("phase_deviation", 0.002, False),
        ("phase_deviation", 0.020, False),
        ("phase_deviation", 0.100, False),
        ("phase_deviation", 1.000, False),
        ("phase_deviation", 1.000, True),
        ("phase_deviation", 15.000, True),
    ]

    jobs = [(15.0, fam, d, c) for (fam, d, c) in variants]

    with Pool(args.procs, maxtasksperchild=1) as pool:
        rows = pool.map(run_cell, jobs, chunksize=1)

    print("\nResults for command period T = 15.0 s:")
    print("  %-28s %5s %11s %9s %6s %10s %7s %6s"
          % ("variant", "n", "advantage", "worst", "pos", "muD lead", "det", "FA"))
    for r in rows:
        if r.get("unformable"):
            print("  %-28s  -- UNFORMABLE (DFT bin spacing > band) --" % r["variant"])
            continue
        f = lambda v, q="%+.2f": (q % v) if v is not None else "--"
        print("  %-28s %5d %11s %9s %3d/%-2d %10s %6.0f%% %5.0f%%"
              % (r["variant"], r["n"], f(r.get("adv_median")), f(r.get("worst"), "%+.1f"),
                 r.get("n_pos", 0), r.get("n", 0), f(r.get("muD_lead_ms")),
                 100 * (r.get("det") or 0), 100 * (r.get("fa") or 0)))

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump(dict(snr_db=SNR, sigma=SIGMA, seeds=SEEDS, reps=REPS, amp_factor=AMP,
                   period_s=15.0, rows=rows), open(args.json, "w"),
              indent=1, default=float)
    print("\nwrote %s" % args.json)

if __name__ == "__main__":
    main()
