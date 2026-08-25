"""experiments/dphi_sweep.py -- detection against the fin excursion, swept over trajectories.

WHY THIS EXISTS. Detection of the fin cue is governed by peak |dphi/dt| -- the fin excursion --
and not by SNR, so a sweep over noise draws on a single trajectory measures nothing about how the
claim generalises. The design below sweeps the quantity that actually moves detection, over
independently drawn trajectories:

    sweep peak |dphi/dt| as the independent variable -- it is settable via fin arm, altitude, Mach
    and command -- not SNR; both muD arms on the PHASE channel; budget recomputed per condition;
    dwell selected by max detection s.t. no-cue FA <= 5%; the kinematic comparator in the same
    table; and the delayed-fin shift test as attribution control.

Each clause is load-bearing. Peak |dphi/dt| is the independent variable because a single flight
condition would select the cue amplitude that flatters the detector. Both muD arms read the PHASE
channel because `slow_time_return` renders the cue as pure phase modulation -- |s| is constant to
4.4e-16, so a statistic reading np.abs(s) sees a channel with no cue in it. The budget is
recomputed per condition because it ranges over 30.2-53.7 ms across flight conditions, and a
"% of budget" divided by a fixed denominator is not that quantity. The dwell is selected subject to
a no-cue false-alarm bound because a cruise-maximum threshold alone is not a zero-false-alarm
operating point. The kinematic comparator shares the table because both arms see the same airframe
delay, so a muD "% of budget" quoted alone measures that delay rather than the radar. The
delayed-fin shift test is the attribution control: if the alarm time does not move when the fin
history moves, the detector is not responding to the fin.

WHAT COUNTS AS A RESULT. Whether a detection threshold in peak |dphi/dt| falls out -- a DESIGN law
of the form "a body-state signature cue is convertible above X rad/s and not below". That is a
statement about when the modality can help, which no amount of single-condition work can supply.

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
# Mach (how much fin is needed for that g). They are swept together on purpose: the design question
# is not "what does one knob do" but "over the envelope this airframe actually flies, where is the
# cue convertible".
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

# The flight-condition grid confounds three things at once: it moves peak |dphi/dt|, but it also
# moves the budget and the airframe response. To isolate |dphi/dt| as the governing variable, the
# ladder below varies ONLY the scatterer lever arm on one fixed trajectory. The airframe, the
# command, the budget and the kinematic comparator are then bit-identical across the ladder and
# peak |dphi/dt| scales exactly linearly with the arm, so any threshold that appears is a property
# of the cue amplitude and of nothing else.
LADDER_ARMS = (0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.30, 0.40)
LADDER_COND = (25000, 7.5, 9.1)

DWELLS = (0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500)
FA_MAX = 0.05                 # the no-cue alarm rate a selected dwell may not exceed
T_CMD = 1.0

# A cell only counts as CONVERSION if it clears all three. Clearing the false-alarm constraint
# alone is not enough: a 3%-detection cell whose median lead is -222.9 ms is a single late alarm at
# the end of the search window, and it must not sit in the same column as a 100%-detection cell at
# +32.4 ms. The criterion is named here so it cannot drift between the sweep and the ladder.
DET_MIN = 0.50                # a coin-flip detector is not a detector
LEAD_MIN_MS = 0.0             # a "lead" that is negative is a detection delay


def converts(cell):
    return (cell is not None and cell.get("med") is not None
            and cell["fa"] <= FA_MAX and cell["det"] >= DET_MIN and cell["med"] > LEAD_MIN_MS)


def return_from_fin(t_fin, delta, snr_db, seed, fin_arm, shift_s=0.0, cue_on=True):
    """Slow-time return whose phase follows the ACTUAL fin history, with two switches.

    `cue_on=False` gives the NULL: identical noise, identical length, identical everything, and no
    fin motion at all. Without this the detection rates below are alarm rates and cannot be told
    apart from noise: a cell can read "100% detection" while alarming on half of pure noise.

    `shift_s` delays the fin history by a fixed amount without touching anything else, so the
    attribution control can ask whether the alarm moves with the fin.
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
    """Instantaneous Doppler |d(phase)/dt| in rad/s -- the PHASE channel. Both muD statistics below
    read this, so they differ only in time-frequency selectivity and not in which channel they see.
    A comparison whose arms differ in channel as well cannot attribute the difference to either."""
    ph = np.unwrap(np.angle(s))
    return np.abs(np.diff(ph, prepend=ph[0])) / DT_R


def stat_matched_phase(t, s, dwell_s):
    """Trailing-mean |dphi/dt|^2 -- matched to a brief broadband velocity burst."""
    n = max(2, int(dwell_s / DT_R))
    x = dphi(s) ** 2
    return np.convolve(x, np.ones(n) / n, mode="full")[:len(t)]


def stat_band_phase(t, s, dwell_s, lo=3.0, hi=10.0):
    """The paper's band-power statistic, moved onto the PHASE channel and made causal.

    Note the constraint that makes this comparison honest: the DFT bin spacing over a dwell T is
    1/T, so a [3,10] Hz band contains ZERO bins until T >= ~100 ms (500 Hz spacing at 2 ms, 100 Hz
    at 10 ms). Where the band is empty the statistic cannot be FORMED, and that is reported as
    'unformable' rather than as a detection failure -- they are different facts.
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
    """The deployable comparator under the identical causal rule: trailing-mean |a_z| on noisy
    measurements. It belongs in this table because both arms see the same airframe delay, so only
    the DIFFERENCE between them is a statement about modality."""
    n = max(2, int(dwell_s / DT_R))
    rk = np.random.default_rng(seed + 991)
    azr = np.interp(t, tf, az) + rk.normal(0, kin_noise, len(t))
    return np.convolve(np.abs(azr), np.ones(n) / n, mode="full")[:len(t)]


def budget_of(fl, t_cmd):
    """Per-condition information budget: first fin motion (2% of its own 99th percentile) to the
    sustained ONSET_G crossing of |a_z|. Recomputed for EVERY condition -- it is not a constant of
    the study, and a lead expressed as '% of budget' against the wrong denominator is meaningless."""
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
    """Advantage as a PAIRED per-seed difference, with an n, an interval and a test.

    The muD and kinematic arms are scored on the SAME noise realisation, so the difference is
    paired and the pairing is the point: both arms see the same airframe, so only the within-seed
    difference is a statement about modality. A pair survives only where BOTH arms alarm, and the
    surviving n is reported because that censoring is informative -- it is the same problem the
    manuscript flags in the IMM arm of Table I.

    Reporting a bare median here would be Sec. IV-5 of this paper's own manuscript, committed in
    the table meant to answer it.
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
    """Detection rate on signal, alarm rate on the NULL, and median lead over detections.

    Both rates come from the same threshold rule (cruise maximum) applied to the same statistic on
    the same seeds -- the null differs only in whether the fin moved. A 'detection rate' quoted
    without its null is an alarm rate, which is how a statistic that fires on 50% of pure noise got
    written down as 100% detection.
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
        # NULL: same seed, same noise, fin held still. Threshold re-estimated from ITS OWN cruise,
        # which is the operating point a fielded detector would actually run at.
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
    """Best (statistic, dwell) by max detection subject to no-cue FA <= FA_MAX, plus the best cell
    ignoring the FA constraint. Returning both matters: 'no detection' and 'detects but alarms on
    noise' are different failures, and collapsing them is how a 50%-false-alarm cell was once
    written down as a success."""
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
    """Vary ONLY the radar lever arm on one fixed trajectory."""
    alt, mach, a_cmd = LADDER_COND
    fl = fly(a_cmd, T_CMD, T_CMD + 0.9, alt=alt, mach=mach)
    tf, delta, az = fl["t"], fl["delta"], fl["az"]
    t_on, budget_ms, _ = budget_of(fl, T_CMD)
    print("\nLEVER-ARM LADDER -- one airframe, one command, one budget (%.1f ms), one comparator."
          % budget_ms)
    print("  %.0f km / M%.1f / %.1f g, fin excursion %.2f deg. Only the scatterer lever arm moves,"
          % (alt / 1000, mach, a_cmd, np.degrees(np.abs(delta).max())))
    print("  so peak |dphi/dt| scales linearly with it and NOTHING else differs between rows.\n")
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
                    help="lever-arm ladder on one fixed trajectory: isolates peak |dphi/dt| from "
                         "the budget and airframe response, which the flight-condition grid "
                         "confounds")
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

    print("PEAK |dphi/dt| SWEEP -- the design law, or the epitaph")
    print("  SNR %.0f dB, %d noise realisations per cell, PRF %.0f Hz" % (args.snr, args.reps, PRF))
    print("  dwell selected by MAX DETECTION subject to no-cue false alarm <= %.0f%%" % (100 * FA_MAX))
    print("  budget recomputed per condition; kinematic comparator scored by the identical rule\n")

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

        # peak |dphi/dt| is a property of the noiseless cue, so measure it noiselessly
        t0 = np.arange(tf[0], tf[-1], DT_R)
        d0 = np.interp(t0, tf, delta)
        peak = float(np.abs(np.diff(4.0 * np.pi * (arm * np.sin(d0)) / (C / FC_HZ),
                                    prepend=0.0) / DT_R).max())
        fin_deg = float(np.degrees(np.abs(delta).max()))

        # ---- muD arm: pick the dwell honestly -------------------------------------------------
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
            # Say WHY it failed. "No detection" and "detects at a 43% false-alarm rate" are
            # different facts and only one of them is a property of the modality.
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

    # ---- is there a threshold in peak |dphi/dt|? ----------------------------------------------
    print("\nDETECTION vs PEAK |dphi/dt| -- is there a design threshold?")
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
            print("  NOT monotone: a failing condition sits above a converting one, so peak")
            print("  |dphi/dt| alone does not order this grid and the budget/airframe confound.")
    elif conv and not dead:
        print("\n  every condition tested converts; no lower threshold is bracketed by this grid")
    else:
        print("\n  NO condition converts. The constructive claim stays dead.")

    # ---- attribution control -----------------------------------------------------------------
    print("\nATTRIBUTION CONTROL -- does the alarm follow the fin, or something else?")
    # Test the most reliably converting condition, not merely the first in peak order: a 3%
    # detection cell has nothing to attribute and its shift slope is noise.
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
