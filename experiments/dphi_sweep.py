"""experiments/dphi_sweep.py -- fin-cue detection as a function of peak |dphi/dt|.

Detection of the fin cue is governed by peak |dphi/dt|, the fin excursion expressed as a phase rate.
This script sweeps peak |dphi/dt| over flight conditions that set it through fin arm, altitude,
Mach and commanded g. For each condition:

  - both muD statistics read the phase channel. `slow_time_return` (causal_dwell_test.py) renders
    the cue as pure phase modulation, with |s| constant to 4.4e-16, so np.abs(s) carries no cue;
  - the information budget is recomputed per condition (30.2-53.7 ms across flight conditions),
    and "% of budget" uses that condition's budget;
  - the dwell is selected by maximum detection subject to a no-cue false-alarm rate <= 5%, since a
    cruise-maximum threshold alone does not give zero false alarms;
  - the kinematic comparator is scored in the same table with the same causal rule. Both arms see
    the same airframe delay, so their difference isolates the sensing modality;
  - a delayed-fin shift test checks attribution: an alarm driven by the fin moves with the fin
    history.

The result is the peak |dphi/dt| above which a body-state signature cue converts to a positive
lead, if the grid brackets one. `--ladder` varies only the scatterer lever arm on one trajectory.

    python experiments/dphi_sweep.py
    python experiments/dphi_sweep.py --reps 40 --json runs/ml/dphi_sweep.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from experiments.physical_cue_test import fly                                 # noqa: E402
from experiments.causal_dwell_test import (                                   # noqa: E402
    PRF, DT_R, ONSET_G, FIN_ARM_M, C, FC_HZ, causal_lead, thr_from_cruise,
)

# Conditions spanning the fin-excursion range. Peak |dphi/dt| is set jointly by the fin arm
# (geometry), the commanded g (how far the fin has to move), and dynamic pressure via altitude and
# Mach (how much fin is needed for that g). The grid covers the envelope this airframe flies; the
# lever-arm ladder below varies a single parameter.
CONDITIONS = [
    # label                     alt_m   mach  a_cmd_g  fin_arm_m
    ("30 km / M7.5 / 9.1 g",    30000,  7.5,     9.1,   0.30),
    ("25 km / M7.5 / 9.1 g",    25000,  7.5,     9.1,   0.30),
    ("25 km / M7.5 / 5.0 g",    25000,  7.5,     5.0,   0.30),
    ("20 km / M7.5 / 9.1 g",    20000,  7.5,     9.1,   0.30),
    ("25 km / M5.0 / 9.1 g",    25000,  5.0,     9.1,   0.30),
    ("15 km / M5.0 / 9.1 g",    15000,  5.0,     9.1,   0.30),
    ("25 km / M7.5 / 2.0 g",    25000,  7.5,     2.0,   0.30),
    ("10 km / M3.0 / 9.1 g",    10000,  3.0,     9.1,   0.30),
    ("25 km / M7.5 / 9.1 g, short arm", 25000, 7.5, 9.1, 0.10),
    ("25 km / M7.5 / 9.1 g, long arm",  25000, 7.5, 9.1, 0.60),
]

# Across the flight-condition grid, peak |dphi/dt|, the budget and the airframe response all vary
# together. The ladder varies only the scatterer lever arm on one fixed trajectory, so the airframe,
# command, budget and kinematic comparator are bit-identical across rungs and peak |dphi/dt| scales
# linearly with the arm. A threshold on the ladder therefore depends on cue amplitude alone.
LADDER_ARMS = (0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.30, 0.40)
LADDER_COND = (25000, 7.5, 9.1)

DWELLS = (0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500)
FA_MAX = 0.05                 # the no-cue alarm rate a selected dwell may not exceed
T_CMD = 1.0

# A cell converts when it meets all three criteria: FA <= FA_MAX, detection >= DET_MIN and median
# lead > LEAD_MIN_MS. The detection and lead floors exclude cells such as 3% detection at a median
# lead of -222.9 ms, which is a single late alarm at the end of the search window. converts() applies
# the same criterion to the sweep and the ladder.
DET_MIN = 0.50                # minimum detection rate on signal
LEAD_MIN_MS = 0.0             # a negative lead is a detection delay


def converts(cell):
    return (cell is not None and cell.get("med") is not None
            and cell["fa"] <= FA_MAX and cell["det"] >= DET_MIN and cell["med"] > LEAD_MIN_MS)


def return_from_fin(t_fin, delta, snr_db, seed, fin_arm, shift_s=0.0, cue_on=True):
    """Slow-time return whose phase follows the fin history delta(t_fin). Returns (t, s).

    `cue_on=False` gives the null: the same noise draw and length with no fin motion, used to
    measure the false-alarm rate.

    `shift_s` delays the fin history by a fixed amount with everything else unchanged, for the
    attribution control.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(t_fin[0], t_fin[-1], DT_R)
    if cue_on:
        d = np.interp(t - shift_s, t_fin, delta, left=delta[0], right=delta[-1])
    else:
        d = np.zeros_like(t)
    lam = C / FC_HZ
    phase = 4.0 * np.pi * (fin_arm * np.sin(d)) / lam
    s = np.exp(1j * phase)
    p_n = 1.0 / (10.0 ** (snr_db / 10.0))
    s = s + np.sqrt(p_n / 2.0) * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))
    return t, s


def dphi(s):
    """Instantaneous Doppler |d(phase)/dt| in rad/s (the phase channel). Both muD statistics below
    read this signal, so they differ only in time-frequency selectivity."""
    ph = np.unwrap(np.angle(s))
    return np.abs(np.diff(ph, prepend=ph[0])) / DT_R


def stat_matched_phase(t, s, dwell_s):
    """Trailing mean of |dphi/dt|^2, matched to a brief broadband velocity burst."""
    n = max(2, int(dwell_s / DT_R))
    x = dphi(s) ** 2
    return np.convolve(x, np.ones(n) / n, mode="full")[:len(t)]


def stat_band_phase(t, s, dwell_s, lo=3.0, hi=10.0):
    """The paper's band-power statistic, computed causally on the phase channel.

    The DFT bin spacing over a dwell T is 1/T, so a [lo, hi] = [3, 10] Hz band contains no bins
    until T >= ~100 ms (500 Hz spacing at 2 ms, 100 Hz at 10 ms). Where the band is empty the
    function returns None and the cell is reported as 'unformable'.
    """
    n = max(2, int(dwell_s / DT_R))
    f = np.fft.rfftfreq(n, d=DT_R)
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return None
    x = dphi(s)
    out = np.full(len(t), np.nan)
    for i in range(n, len(t)):
        w = x[i - n:i]
        w = w - w.mean()
        out[i] = (np.abs(np.fft.rfft(w)) ** 2)[band].sum() / n
    return out


def stat_kinematic(t, tf, az, dwell_s, seed, kin_noise):
    """Kinematic comparator under the same causal rule: trailing-mean |a_z| on measurements with
    Gaussian noise of standard deviation kin_noise. Both arms see the same airframe delay, so the
    difference between them measures the sensing modality."""
    n = max(2, int(dwell_s / DT_R))
    rk = np.random.default_rng(seed + 991)
    azr = np.interp(t, tf, az) + rk.normal(0, kin_noise, len(t))
    return np.convolve(np.abs(azr), np.ones(n) / n, mode="full")[:len(t)]


def budget_of(fl, t_cmd):
    """Per-condition information budget: first fin motion (2% of the 99th percentile of |delta|) to
    the first ONSET_G crossing of |a_z| sustained for 5 ms after t_cmd. Returns (t_on, budget_ms,
    t_fin), or (None, None, None) if either event is absent. Computed separately for each condition."""
    t, delta, az = fl["t"], fl["delta"], fl["az"]
    ref = np.percentile(np.abs(delta), 99)
    if ref <= 0:
        return None, None, None
    t_fin = float(t[int(np.argmax(np.abs(delta) >= 0.02 * ref))])
    ab = np.abs(az) > ONSET_G
    need = int(0.005 / 1e-4)
    idx = [i for i in range(int(t_cmd / 1e-4), len(t) - need) if ab[i:i + need].all()]
    if not idx:
        return None, None, None
    t_on = float(t[idx[0]])
    return t_on, 1000.0 * (t_on - t_fin), t_fin


def paired_advantage(make_stat, tf, delta, az, t_on, snr, reps, fin_arm, kin_noise, kin_dwell):
    """Paired per-seed advantage (muD lead minus kinematic lead, ms) with n, interval and test.

    The muD and kinematic arms are scored on the same noise realisation, so the per-seed difference
    is paired. A pair is kept only where both arms alarm, and the surviving n is reported because
    that censoring is informative. Returns dict(n, med, lo, hi, p): median, 95% bootstrap interval
    (4000 resamples) and two-sided Wilcoxon signed-rank p-value; med/lo/hi/p are None when fewer
    than 3 pairs survive. Returns None if the muD statistic is unformable.
    """
    from scipy.stats import wilcoxon
    diffs = []
    for r in range(reps):
        t, s = return_from_fin(tf, delta, snr, 4000 + r, fin_arm)
        st = make_stat(t, s)
        if st is None:
            return None
        lm = causal_lead(t, st, thr_from_cruise(t, st, t_on), t_on)
        stk = stat_kinematic(t, tf, az, kin_dwell, 4000 + r, kin_noise)
        lk = causal_lead(t, stk, thr_from_cruise(t, stk, t_on), t_on)
        if lm is not None and lk is not None:
            diffs.append(1000.0 * (lm - lk))
    a = np.array(diffs, float)
    if a.size < 3:
        return dict(n=int(a.size), med=None, lo=None, hi=None, p=None)
    rng = np.random.default_rng(20260731)
    bt = np.median(rng.choice(a, (4000, a.size)), axis=1)
    try:
        p = float(wilcoxon(a, alternative="two-sided", zero_method="zsplit").pvalue)
    except ValueError:
        p = float("nan")
    return dict(n=int(a.size), med=float(np.median(a)),
                lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)), p=p)


def score(make_stat, tf, delta, az, t_on, snr, reps, fin_arm, kin_noise, shift_s=0.0):
    """Detection rate on signal, false-alarm rate on the null, and median lead over detections.

    Both rates come from the same threshold rule (cruise maximum) applied to the same statistic on
    the same seeds; the null differs only in that the fin does not move. Returns
    dict(det, fa, med, leads, n) with leads in ms, or None if the statistic is unformable.
    """
    leads, det, fa = [], 0, 0
    n_ok = 0
    for r in range(reps):
        t, s = return_from_fin(tf, delta, snr, 4000 + r, fin_arm, shift_s=shift_s, cue_on=True)
        st = make_stat(t, s)
        if st is None:
            return None
        thr = thr_from_cruise(t, st, t_on)
        n_ok += 1
        lead = causal_lead(t, st, thr, t_on)
        if lead is not None:
            det += 1
            leads.append(1000.0 * lead)
        # Null: same seed and noise, fin held still. The threshold is re-estimated from the null's
        # own cruise segment, which is the operating point a fielded detector runs at.
        tn, sn = return_from_fin(tf, delta, snr, 4000 + r, fin_arm, cue_on=False)
        stn = make_stat(tn, sn)
        if stn is not None and causal_lead(tn, stn, thr_from_cruise(tn, stn, t_on), t_on) is not None:
            fa += 1
    if not n_ok:
        return None
    return dict(det=det / n_ok, fa=fa / n_ok,
                med=float(np.median(leads)) if leads else None,
                leads=leads, n=n_ok)


def best_dwell(tf, delta, az, t_on, arm, args):
    """Best (statistic, dwell) cell by detection, then lead, among cells that pass converts(), plus
    the best cell by detection alone with no constraint. The unconstrained cell distinguishes a
    detector that never alarms from one that detects but also alarms on noise. Returns
    (best, unconstrained, cells)."""
    best, unconstrained, cells = None, None, []
    for dw in DWELLS:
        for sname, maker in (("matched", lambda t, s, d=dw: stat_matched_phase(t, s, d)),
                             ("band 3-10", lambda t, s, d=dw: stat_band_phase(t, s, d))):
            res = score(maker, tf, delta, az, t_on, args.snr, args.reps, arm, args.kin_noise)
            if res is None:
                cells.append(dict(dwell=dw, stat=sname, unformable=True))
                continue
            cells.append(dict(dwell=dw, stat=sname, det=res["det"], fa=res["fa"], med=res["med"]))
            cand = dict(dwell=dw, stat=sname, **{k: res[k] for k in ("det", "fa", "med", "n")})
            if unconstrained is None or res["det"] > unconstrained["det"]:
                unconstrained = cand
            if converts(cand):
                if best is None or (cand["det"], cand["med"]) > (best["det"], best["med"]):
                    best = cand
    return best, unconstrained, cells


def run_ladder(args):
    """Vary only the scatterer lever arm on one fixed trajectory (LADDER_COND). Returns the rows."""
    alt, mach, a_cmd = LADDER_COND
    fl = fly(a_cmd, T_CMD, T_CMD + 0.9, alt=alt, mach=mach)
    tf, delta, az = fl["t"], fl["delta"], fl["az"]
    t_on, budget_ms, _ = budget_of(fl, T_CMD)
    print("\nLEVER-ARM LADDER -- one airframe, one command, one budget (%.1f ms), one comparator."
          % budget_ms)
    print("  %.0f km / M%.1f / %.1f g, fin excursion %.2f deg. Only the scatterer lever arm moves,"
          % (alt / 1000, mach, a_cmd, np.degrees(np.abs(delta).max())))
    print("  so peak |dphi/dt| scales linearly with it and all other inputs are identical across rows.\n")
    print("  %-8s %10s %-16s %7s %7s %9s %9s   %s"
          % ("arm (m)", "peak dphi", "best dwell", "det", "FA", "lead ms", "% budget",
             "unconstrained best"))
    print("  " + "-" * 108)
    rows = []
    for arm in LADDER_ARMS:
        t0 = np.arange(tf[0], tf[-1], DT_R)
        d0 = np.interp(t0, tf, delta)
        peak = float(np.abs(np.diff(4.0 * np.pi * (arm * np.sin(d0)) / (C / FC_HZ),
                                    prepend=0.0) / DT_R).max())
        best, unc, _ = best_dwell(tf, delta, az, t_on, arm, args)
        u = ("det %.0f%% @ FA %.0f%%" % (100 * unc["det"], 100 * unc["fa"])) if unc else "--"
        if best is None:
            print("  %-8.2f %10.0f %-16s %7s %7s %9s %9s   %s"
                  % (arm, peak, "none survives", "--", "--", "--", "--", u))
            rows.append(dict(arm=arm, peak=peak, converts=False,
                             unconstrained=unc and {k: unc[k] for k in ("det", "fa", "med")}))
        else:
            print("  %-8.2f %10.0f %-16s %6.0f%% %6.0f%% %+9.1f %8.0f%%   %s"
                  % (arm, peak, "%s @ %.0f ms" % (best["stat"], 1000 * best["dwell"]),
                     100 * best["det"], 100 * best["fa"], best["med"],
                     100 * best["med"] / budget_ms, u))
            rows.append(dict(arm=arm, peak=peak, converts=True,
                             **{k: best[k] for k in ("dwell", "stat", "det", "fa", "med")}))
    yes = [r for r in rows if r["converts"]]
    no = [r for r in rows if not r["converts"]]
    if yes and no:
        print("\n  THRESHOLD BRACKETED: converts at >= %.0f rad/s, fails at <= %.0f rad/s"
              % (min(r["peak"] for r in yes), max(r["peak"] for r in no)))
        print("  The budget, airframe and comparator are identical across every row, so the")
        print("  transition is a property of cue amplitude alone.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--kin-noise", type=float, default=0.3)
    ap.add_argument("--ladder", action="store_true",
                    help="lever-arm ladder on one fixed trajectory: varies peak |dphi/dt| with "
                         "the budget and airframe response held fixed (they co-vary across the "
                         "flight-condition grid)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if args.ladder:
        rows = run_ladder(args)
        if args.json:
            os.makedirs(os.path.dirname(args.json), exist_ok=True)
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1, default=float)
            print("\nwrote %s" % args.json)
        return

    print("Peak |dphi/dt| sweep over flight conditions")
    print("  SNR %.0f dB, %d noise realisations per cell, PRF %.0f Hz" % (args.snr, args.reps, PRF))
    print("  dwell selected by MAX DETECTION subject to no-cue false alarm <= %.0f%%" % (100 * FA_MAX))
    print("  budget recomputed per condition; kinematic comparator scored by the same rule\n")

    out = []
    print("%-34s %9s %9s %8s   %-22s %8s %8s %8s   %8s %9s"
          % ("condition", "peak dphi", "fin deg", "budget", "best muD dwell", "det", "FA", "lead",
             "kin lead", "ADVANTAGE"))
    print("-" * 146)

    for label, alt, mach, a_cmd, arm in CONDITIONS:
        fl = fly(a_cmd, T_CMD, T_CMD + 0.9, alt=alt, mach=mach)
        tf, delta, az = fl["t"], fl["delta"], fl["az"]
        t_on, budget_ms, t_fin = budget_of(fl, T_CMD)
        if t_on is None:
            print("%-34s %9s  (never reaches the %.1f m/s^2 onset threshold -- excluded)"
                  % (label, "--", ONSET_G))
            out.append(dict(label=label, alt=alt, mach=mach, a_cmd=a_cmd, arm=arm, excluded=True))
            continue

        # peak |dphi/dt| of the noiseless cue
        t0 = np.arange(tf[0], tf[-1], DT_R)
        d0 = np.interp(t0, tf, delta)
        peak = float(np.abs(np.diff(4.0 * np.pi * (arm * np.sin(d0)) / (C / FC_HZ),
                                    prepend=0.0) / DT_R).max())
        fin_deg = float(np.degrees(np.abs(delta).max()))

        # ---- muD arm: dwell selection ---------------------------------------------------------
        best, unc, percell = best_dwell(tf, delta, az, t_on, arm, args)

        # ---- kinematic comparator: same rule, same selection ----------------------------------
        kbest = None
        for dw in DWELLS:
            kl, kdet, kfa, nk = [], 0, 0, 0
            for r in range(args.reps):
                t, s = return_from_fin(tf, delta, args.snr, 4000 + r, arm)
                stk = stat_kinematic(t, tf, az, dw, 4000 + r, args.kin_noise)
                nk += 1
                lk = causal_lead(t, stk, thr_from_cruise(t, stk, t_on), t_on)
                if lk is not None:
                    kdet += 1
                    kl.append(1000.0 * lk)
                azf = np.zeros_like(az)          # null: no maneuver, same measurement noise
                stkn = stat_kinematic(t, tf, azf, dw, 4000 + r, args.kin_noise)
                if causal_lead(t, stkn, thr_from_cruise(t, stkn, t_on), t_on) is not None:
                    kfa += 1
            if kl and kfa / max(nk, 1) <= FA_MAX:
                key = (kdet / nk, float(np.median(kl)))
                if kbest is None or key > (kbest["det"], kbest["med"]):
                    kbest = dict(dwell=dw, det=kdet / nk, fa=kfa / max(nk, 1),
                                 med=float(np.median(kl)), leads=kl)

        if best is None:
            # Report the best unconstrained cell, which separates "no detection" from detection
            # at a high false-alarm rate.
            why = ("no cell converts" if unc is None
                   else "best: det %.0f%% @ FA %.0f%%" % (100 * unc["det"], 100 * unc["fa"]))
            print("%-34s %9.0f %9.2f %7.1fms   %-22s %8s %8s %8s   %8s %9s   %s"
                  % (label, peak, fin_deg, budget_ms, "NONE converts", "--", "--", "--",
                     ("%+.1f" % kbest["med"]) if kbest else "--", "--", why))
            out.append(dict(label=label, alt=alt, mach=mach, a_cmd=a_cmd, arm=arm, peak=peak,
                            fin_deg=fin_deg, budget_ms=budget_ms, best=None,
                            unconstrained=unc and {k: unc[k] for k in ("det", "fa", "med")},
                            kin=kbest and {k: kbest[k] for k in ("dwell", "det", "fa", "med")},
                            cells=percell))
            continue

        maker = ((lambda t, s, d=best["dwell"]: stat_matched_phase(t, s, d))
                 if best["stat"] == "matched"
                 else (lambda t, s, d=best["dwell"]: stat_band_phase(t, s, d)))
        pa = (paired_advantage(maker, tf, delta, az, t_on, args.snr, args.reps, arm,
                               args.kin_noise, kbest["dwell"]) if kbest else None)
        adv_s = ("%+.1f [%+.1f,%+.1f] n=%d p=%.3f"
                 % (pa["med"], pa["lo"], pa["hi"], pa["n"], pa["p"])
                 if pa and pa["med"] is not None else "--")
        print("%-34s %9.0f %9.2f %7.1fms   %-22s %7.0f%% %7.0f%% %+7.1f   %8s   %s"
              % (label, peak, fin_deg, budget_ms,
                 "%s @ %.0f ms" % (best["stat"], 1000 * best["dwell"]),
                 100 * best["det"], 100 * best["fa"], best["med"],
                 ("%+.1f" % kbest["med"]) if kbest else "--", adv_s))
        out.append(dict(label=label, alt=alt, mach=mach, a_cmd=a_cmd, arm=arm, peak=peak,
                        fin_deg=fin_deg, budget_ms=budget_ms,
                        best={k: best[k] for k in ("dwell", "stat", "det", "fa", "med")},
                        kin=kbest and {k: kbest[k] for k in ("dwell", "det", "fa", "med")},
                        advantage=pa, cells=percell))

    # ---- detection vs peak |dphi/dt| ----------------------------------------------------------
    print("\nDetection vs peak |dphi/dt| (sorted by peak)")
    live = [o for o in out if not o.get("excluded") and o.get("peak") is not None]
    live.sort(key=lambda o: o["peak"])
    print("%-34s %10s %10s %10s %10s" % ("condition", "peak dphi", "det", "lead ms", "% budget"))
    print("-" * 78)
    for o in live:
        b = o.get("best")
        if b is None:
            u = o.get("unconstrained")
            print("%-34s %10.0f %10s %10s %10s   %s"
                  % (o["label"], o["peak"], "none", "--", "--",
                     "" if not u else "(unconstrained: det %.0f%% at FA %.0f%%)"
                     % (100 * u["det"], 100 * u["fa"])))
        else:
            print("%-34s %10.0f %9.0f%% %+10.1f %9.0f%%"
                  % (o["label"], o["peak"], 100 * b["det"], b["med"],
                     100 * b["med"] / o["budget_ms"]))

    conv = [o for o in live if o.get("best")]
    dead = [o for o in live if not o.get("best")]
    if conv and dead:
        lo_ok, hi_bad = min(o["peak"] for o in conv), max(o["peak"] for o in dead)
        print("\n  converts (det>=%.0f%%, FA<=%.0f%%, lead>0) at >= %.0f rad/s; fails at <= %.0f rad/s"
              % (100 * DET_MIN, 100 * FA_MAX, lo_ok, hi_bad))
        if hi_bad < lo_ok:
            print("  threshold bracketed in [%.0f, %.0f] rad/s, monotone across this grid."
                  % (hi_bad, lo_ok))
        else:
            print("  Not monotone: a failing condition sits above a converting one, so peak")
            print("  |dphi/dt| alone does not order this grid; budget and airframe response also vary.")
    elif conv and not dead:
        print("\n  every condition tested converts; no lower threshold is bracketed by this grid")
    else:
        print("\n  No condition converts.")

    # ---- attribution control -----------------------------------------------------------------
    print("\nAttribution control: alarm time vs fin-history shift")
    # Use the converting condition with the highest detection rate (ties broken by lower peak); a
    # low-detection cell gives a noisy shift slope.
    ref = max((o for o in live if o.get("best")),
              key=lambda o: (o["best"]["det"], -o["peak"]), default=None)
    if ref is None:
        print("  no converting condition to test.")
    else:
        alt, mach, a_cmd, arm = ref["alt"], ref["mach"], ref["a_cmd"], ref["arm"]
        fl = fly(a_cmd, T_CMD, T_CMD + 0.9, alt=alt, mach=mach)
        tf, delta, az = fl["t"], fl["delta"], fl["az"]
        t_on, budget_ms, _ = budget_of(fl, T_CMD)
        dw, sname = ref["best"]["dwell"], ref["best"]["stat"]
        maker = ((lambda t, s: stat_matched_phase(t, s, dw)) if sname == "matched"
                 else (lambda t, s: stat_band_phase(t, s, dw)))
        print("  %s | %s @ %.0f ms | if the alarm tracks the fin, lead falls 1:1 with the shift"
              % (ref["label"], sname, 1000 * dw))
        print("  %-12s %12s %12s" % ("fin shift", "median lead", "delta vs 0"))
        base = None
        shifts = []
        for sh in (0.000, 0.006, 0.012, 0.024):
            res = score(maker, tf, delta, az, t_on, args.snr, args.reps, arm, args.kin_noise,
                        shift_s=sh)
            if res is None or res["med"] is None:
                print("  %-12.0f %12s" % (1000 * sh, "no detection"))
                continue
            if base is None:
                base = res["med"]
            shifts.append((1000 * sh, res["med"]))
            print("  %-12.0f %+12.1f %+12.1f" % (1000 * sh, res["med"], res["med"] - base))
        if len(shifts) >= 3:
            x = np.array([s[0] for s in shifts])
            y = np.array([s[1] for s in shifts])
            A = np.vstack([x, np.ones_like(x)]).T
            sl, ic = np.linalg.lstsq(A, y, rcond=None)[0]
            print("  slope %+.3f ms of lead per ms of fin shift (-1.000 = the alarm IS the fin)"
                  % sl)
            out.append(dict(attribution_slope=float(sl)))

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
