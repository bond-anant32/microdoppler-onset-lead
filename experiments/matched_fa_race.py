"""experiments/matched_fa_race.py -- lead advantage with both arms at a matched false-alarm rate.

Under the trailing-window threshold rule, online_fa_rate.py measures these quiescent false-alarm
rates over 1782 s of flight per arm:

    muD    0.0146 /s   (one per 68.5 s)
    CUSUM  0.4394 /s   (one per 2.3 s)

so a common threshold rule runs CUSUM at about 30x the muD arm's rate. This script calibrates a
threshold scale c for each arm separately on quiescent records, targeting a common alarm-event rate,
and then races the arms on the manoeuvre records at that rate:

    eta_arm(t) = c_arm * max{ z(t') : t' in [t-0.60, t-0.12] }

The manoeuvre records play no part in setting c. calibrate() returns the achieved rate with each
scale, and the matched-rate summary includes only targets at which both arms achieve the same rate.
With c = 1 for both arms the paired advantage is the reported +17.0 ms.

    python experiments/matched_fa_race.py --json runs/ml/matched_fa_race.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from scipy.stats import wilcoxon                                             # noqa: E402
import experiments.multiclass_lead as ml                                      # noqa: E402
from experiments.multiclass_lead import class_windows                         # noqa: E402
from experiments.dphi_sweep import return_from_fin, stat_matched_phase        # noqa: E402
from experiments.causal_threshold_test import rolling_thr                     # noqa: E402
from experiments.online_fa_rate import count_alarm_events                     # noqa: E402
from experiments.causal_dwell_test import DT_R, DECISION_GRID_S               # noqa: E402

AMP, SNR, REPS, DWELL, SIGMA, SEEDS = 0.2798, 40.0, 12, 0.002, 0.3, 30
CAL_DUR_S, CAL_REC = 60.0, 12          # quiescent calibration records, separate from the race
TARGETS = (1e-1, 1e-2, 1e-3)           # common false-alarm rates, alarms per second
NEED = 3


def quiescent_stats(arm, n_rec, dur):
    """Calibration set: (t, z, rolling threshold) for n_rec quiescent records of dur seconds."""
    out = []
    t_fin = np.arange(0.0, dur, 1e-4)
    still = np.zeros_like(t_fin)
    for k in range(n_rec):
        if arm == "muD":
            t, s = return_from_fin(t_fin, still, SNR, 7000 + k, ml.FIN_ARM_M, cue_on=False)
            z = stat_matched_phase(t, s, DWELL)
        else:
            t = np.arange(t_fin[0], t_fin[-1], DT_R)
            rk = np.random.default_rng(7000 + k + 991)
            z = ml.stat_cusum(np.abs(rk.normal(0, SIGMA, len(t))), DWELL)
        out.append((t, z, rolling_thr(t, z)))
    return out


def rate_at(cal, c):
    """Measured alarm-event rate (per second) over the calibration set at threshold scale c."""
    ev = sec = 0
    for t, z, thr in cal:
        e, s = count_alarm_events(t, z, c * thr)
        ev += e
        sec += s
    return (ev / sec) if sec else float("nan")


def calibrate(cal, target, lo=0.20, hi=200.0, iters=34):
    """Search a threshold scale c in [lo, hi] by bisection toward `target` alarms per second.

    Returns (c, achieved rate). Excursion counts are not necessarily monotone in c, since raising
    a threshold can split one event into two, so the rate actually achieved is measured and
    returned, and comparisons use the achieved rates.
    """
    if rate_at(cal, hi) > target:
        return hi, rate_at(cal, hi)
    if rate_at(cal, lo) < target:
        return lo, rate_at(cal, lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rate_at(cal, mid) > target:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    return c, rate_at(cal, c)


def lead_scaled(t, z, thr, c, t_on, search_pre=0.25):
    """Lead (s) of the first 3-of-3 run above c*thr on the 1 ms grid, or None; as causal_lead."""
    h = c * np.asarray(thr, float)
    m = np.isfinite(z) & np.isfinite(h)
    tt, ss, hh = t[m], np.asarray(z)[m], h[m]
    step = max(1, int(round(DECISION_GRID_S / DT_R)))
    tt, ss, hh = tt[::step], ss[::step], hh[::step]
    w = (tt >= t_on - search_pre) & (tt <= t_on + 0.30)
    tt, ss, hh = tt[w], ss[w], hh[w]
    above = ss > hh
    run = 0
    for i in range(len(above)):
        run = run + 1 if above[i] else 0
        if run >= NEED:
            return float(t_on - tt[i - NEED + 1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("MATCHED-FALSE-ALARM-RATE RACE")
    print("  thresholds calibrated on %d x %.0f s of QUIESCENT flight (held out from the race),"
          % (CAL_REC, CAL_DUR_S))
    print("  then applied to the %d manoeuvre trajectories at %d reps each.\n" % (SEEDS, REPS))

    cal = {a: quiescent_stats(a, CAL_REC, CAL_DUR_S) for a in ("muD", "CUSUM")}
    print("  measured rate at scale c=1, the plain trailing max:")
    for a in ("muD", "CUSUM"):
        print("     %-6s %.4f /s" % (a, rate_at(cal[a], 1.0)))

    # ---- precompute the manoeuvre statistics once ------------------------------------------------
    recs = []
    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        if not wins:
            continue
        per = []
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
            fl = ml.drive_airframe(w2["t"], w2["a_cmd"], w2["V"], w2["alt"])
            t_on, _ = ml.onset_from_achieved(fl["t"], fl["az"])
            if t_on is None:
                continue
            for r in range(REPS):
                t, s = return_from_fin(fl["t"], fl["delta"], SNR, 4000 + r, ml.FIN_ARM_M)
                zm = stat_matched_phase(t, s, DWELL)
                rk = np.random.default_rng(4000 + r + 991)
                azr = np.abs(np.interp(t, fl["t"], fl["az"]) + rk.normal(0, SIGMA, len(t)))
                zk = ml.stat_cusum(azr, DWELL)
                per.append((t, zm, rolling_thr(t, zm), zk, rolling_thr(t, zk), t_on))
        if per:
            recs.append(per)
    print("  %d trajectories x %d reps prepared\n" % (len(recs), REPS))

    print("%12s %9s %9s %6s %10s %8s %6s %8s"
          % ("target /s", "c_muD", "c_CUSUM", "n", "advantage", "worst", "pos", "det"))
    print("-" * 82)
    rows = []
    for tgt in TARGETS:
        c_mu, r_mu = calibrate(cal["muD"], tgt)
        c_ku, r_ku = calibrate(cal["CUSUM"], tgt)
        per_traj, dets = [], []
        for per in recs:
            ev = []
            for t, zm, hm, zk, hk, t_on in per:
                lm = lead_scaled(t, zm, hm, c_mu, t_on)
                lk = lead_scaled(t, zk, hk, c_ku, t_on)
                dets.append(1.0 if lm is not None else 0.0)
                if lm is not None and lk is not None:
                    ev.append(1000 * (lm - lk))
            if ev:
                per_traj.append(float(np.median(ev)))
        a = np.asarray(per_traj, float)
        row = dict(target_per_s=tgt, c_muD=c_mu, c_CUSUM=c_ku,
                   achieved_muD=r_mu, achieved_CUSUM=r_ku, n=int(a.size),
                   adv_median=float(np.median(a)) if a.size else None,
                   worst=float(a.min()) if a.size else None,
                   n_pos=int((a > 0).sum()),
                   det=float(np.mean(dets)) if dets else None,
                   p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None,
                   adv=[float(x) for x in a])
        rows.append(row)
        print("%12.0e %9.2f %9.2f %6d %10s %8s %3d/%-2d %7.0f%%"
              % (tgt, c_mu, c_ku, row["n"],
                 "%+.2f" % row["adv_median"] if row["adv_median"] is not None else "--",
                 "%+.1f" % row["worst"] if row["worst"] is not None else "--",
                 row["n_pos"], row["n"], 100 * (row["det"] or 0)))

    ok = [r for r in rows if r["adv_median"] is not None
          and np.isclose(r["achieved_muD"], r["achieved_CUSUM"], rtol=1e-9, atol=1e-12)]
    if ok:
        print("\nAcross %d rates with equal recorded arm rates, the paired median advantage is %s."
              % (len(ok), "POSITIVE" if all(r["adv_median"] > 0 for r in ok)
                 else "*** NOT ALWAYS POSITIVE ***"))
        print("Reported (unmatched, both arms at the plain trailing max): +17.00 ms.")
    unequal = [r for r in rows if not np.isclose(r["achieved_muD"], r["achieved_CUSUM"],
                                              rtol=1e-9, atol=1e-12)]
    if unequal:
        print("Unequal recorded rates; excluded from the matched-rate summary: %s"
              % ", ".join("%.0e /s target" % r["target_per_s"] for r in unequal))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(sigma=SIGMA, snr_db=SNR, reps=REPS, seeds=SEEDS,
                       cal_records=CAL_REC, cal_dur_s=CAL_DUR_S, rows=rows),
                  open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
