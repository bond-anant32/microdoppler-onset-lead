"""experiments/causal_threshold_test.py -- does the operating point need to know t_on?

THE OBJECTION. The zero-false-alarm threshold is

    eta = max_{t in W} z(t),   W = [t_on - 0.60, t_on - 0.12] s

and W is ANCHORED TO t_on -- the very instant the detector is trying to find. An operational
tracker does not know t_on, so it cannot select a window guaranteed to hold only quiescent cruise.
The charge: the operating point presumes knowledge of where the manoeuvre will NOT occur, so "zero
false alarms" is definitional rather than measured.

Half of that the letter already states: no alarm is POSSIBLE inside W, because eta is the maximum
over W. What was never tested is the half that decides the objection -- whether the anchoring is a
CONVENIENCE or a DEPENDENCE. Three rules, both arms treated identically so the comparison stays
symmetric (a defect in a shared threshold rule shifts both leads and cancels in the paired
advantage; that is why this must be measured rather than argued):

  A  shipped        eta from [t_on-0.60, t_on-0.12].          CONTROL: must reproduce +17.0 ms.
  B  rolling causal eta(t) = max z over the TRAILING window [t-0.60, t-0.12], recomputed at every
                    decision sample, using no knowledge of t_on whatsoever. This is what a tracker
                    could actually run: trailing self-calibration behind a guard band.
  C  anchor error   eta from [t_on+D-0.60, t_on+D-0.12], D = +50, +100, +200 ms, so the window
                    slides PAST the command and is progressively contaminated by manoeuvre samples
                    -- what happens to B if the guard band is too short.

WHAT WOULD FALSIFY THE LETTER. If B collapses, the operating point is an oracle and the headline is
a readout of knowing t_on. If B reproduces A, the anchoring is bookkeeping.

Note on the SEARCH window: causal_lead looks for an alarm in [t_on-search_pre, t_on+0.30]. That
bounds where an alarm is LOOKED FOR, not where one may occur; it is swept separately (search_pre in
sensitivity_sweep) and cannot hide a pre-command alarm here, because this script counts alarms
preceding the command directly and reports them per rule.

    python experiments/causal_threshold_test.py --json runs/ml/causal_threshold.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from scipy.ndimage import maximum_filter1d                                    # noqa: E402
from scipy.stats import wilcoxon                                             # noqa: E402
import experiments.multiclass_lead as ml                                      # noqa: E402
from experiments.multiclass_lead import class_windows                         # noqa: E402
from experiments.dphi_sweep import return_from_fin, stat_matched_phase        # noqa: E402
from experiments.causal_dwell_test import (                                   # noqa: E402
    thr_from_cruise, DT_R, DECISION_GRID_S,
)

AMP, SNR, REPS, DWELL, SIGMA, SEEDS = 0.2798, 40.0, 12, 0.002, 0.3, 30
CRUISE, GUARD = 0.60, 0.12                     # shipped window length and guard band
MODES = [("shipped", 0.0), ("rolling", 0.0),
         ("anchor", 0.05), ("anchor", 0.10), ("anchor", 0.20)]


def rolling_thr(t, stat, lo=CRUISE, hi=GUARD):
    """eta(t) = max of stat over the TRAILING window [t-lo, t-hi]. Uses no t_on.

    O(n) via a sliding-window maximum. Where the window is not yet wholly inside the record the
    threshold is +inf -- no alarm permitted, the conservative choice and the one a real system
    makes while its calibration buffer is still filling.
    """
    n = len(t)
    dt = float(t[1] - t[0])
    ilo, ihi = int(round(lo / dt)), int(round(hi / dt))
    L = ilo - ihi
    if L < 9 or n <= ilo:
        return np.full(n, np.inf)
    # centred sliding max, then shift so the window sits at [i-ilo, i-ihi]
    mx = maximum_filter1d(np.asarray(stat, float), size=L, mode="nearest")
    k = (ilo + ihi) // 2
    out = np.full(n, np.inf)
    out[k:] = mx[:n - k]
    out[:ilo] = np.inf                          # buffer not yet full
    return out


def shifted_thr(t, stat, t_on, shift):
    """The shipped rule with its anchor moved LATE by `shift` seconds."""
    m = np.isfinite(stat) & (t >= t_on + shift - CRUISE) & (t <= t_on + shift - GUARD)
    return float(np.max(stat[m])) if m.sum() > 8 else np.inf


def lead_vs_threshold(t, stat, thr, t_on, need_ms=3.0, search_pre=0.25):
    """causal_lead, but `thr` may be a per-sample ARRAY as well as a scalar.

    Identical decimation, search window and 3-sample run rule as causal_dwell_test.causal_lead, so
    rule B differs from rule A in the THRESHOLD ONLY.
    """
    thr_arr = np.full(len(t), float(thr)) if np.isscalar(thr) else np.asarray(thr, float)
    m = np.isfinite(stat) & np.isfinite(thr_arr)
    tt, ss, hh = t[m], stat[m], thr_arr[m]
    step = max(1, int(round(DECISION_GRID_S / DT_R)))
    tt, ss, hh = tt[::step], ss[::step], hh[::step]
    w = (tt >= t_on - search_pre) & (tt <= t_on + 0.30)
    tt, ss, hh = tt[w], ss[w], hh[w]
    need = max(2, int(need_ms))
    above = ss > hh
    run = 0
    for i in range(len(above)):
        run = run + 1 if above[i] else 0
        if run >= need:
            return float(t_on - tt[i - need + 1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("CAUSAL-THRESHOLD TEST -- does the operating point need to know t_on?")
    print("  n=%d trajectories x %d reps, sigma=%.1f, %.0f dB, amplitude x%.4f\n"
          % (SEEDS, REPS, SIGMA, SNR, AMP))

    acc = {m: dict(traj=[], mud=[], kin=[], det=[], pre=0) for m in MODES}
    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        if not wins:
            continue
        ev = {m: dict(adv=[], mu=[], kin=[]) for m in MODES}
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
            fl = ml.drive_airframe(w2["t"], w2["a_cmd"], w2["V"], w2["alt"])
            t_on, _ = ml.onset_from_achieved(fl["t"], fl["az"])
            if t_on is None:
                continue
            budget_ms = 1000.0 * (t_on - w2["t_cmd"])
            tf, delta, az = fl["t"], fl["delta"], fl["az"]
            for r in range(REPS):
                # statistics computed ONCE per realisation; only the threshold rule varies
                t, s = return_from_fin(tf, delta, SNR, 4000 + r, ml.FIN_ARM_M)
                st = stat_matched_phase(t, s, DWELL)
                rk = np.random.default_rng(4000 + r + 991)
                azr = np.abs(np.interp(t, tf, az) + rk.normal(0, SIGMA, len(t)))
                stk = ml.stat_cusum(azr, DWELL)
                roll_m, roll_k = None, None
                for mode in MODES:
                    name, shift = mode
                    if name == "shipped":
                        hm, hk = thr_from_cruise(t, st, t_on), thr_from_cruise(t, stk, t_on)
                    elif name == "rolling":
                        if roll_m is None:
                            roll_m, roll_k = rolling_thr(t, st), rolling_thr(t, stk)
                        hm, hk = roll_m, roll_k
                    else:
                        hm = shifted_thr(t, st, t_on, shift)
                        hk = shifted_thr(t, stk, t_on, shift)
                    lm = lead_vs_threshold(t, st, hm, t_on)
                    lk = lead_vs_threshold(t, stk, hk, t_on)
                    acc[mode]["det"].append(1.0 if lm is not None else 0.0)
                    if lm is not None:
                        ev[mode]["mu"].append(1000 * lm)
                        if 1000 * lm > budget_ms + 1e-9:
                            acc[mode]["pre"] += 1
                    if lk is not None:
                        ev[mode]["kin"].append(1000 * lk)
                    if lm is not None and lk is not None:
                        ev[mode]["adv"].append(1000 * (lm - lk))
        for mode in MODES:
            if ev[mode]["adv"]:
                acc[mode]["traj"].append(float(np.median(ev[mode]["adv"])))
            if ev[mode]["mu"]:
                acc[mode]["mud"].append(float(np.median(ev[mode]["mu"])))
            if ev[mode]["kin"]:
                acc[mode]["kin"].append(float(np.median(ev[mode]["kin"])))

    print("%-27s %4s %11s %8s %6s %9s %9s %6s %5s"
          % ("threshold rule", "n", "advantage", "worst", "pos", "muD", "CUSUM", "det", "pre"))
    print("-" * 96)
    rows = []
    for mode in MODES:
        name, shift = mode
        a = np.asarray(acc[mode]["traj"], float)
        row = dict(mode=name, shift_ms=1000 * shift, n=int(a.size),
                   adv_median=float(np.median(a)) if a.size else None,
                   n_pos=int((a > 0).sum()), worst=float(a.min()) if a.size else None,
                   muD_lead_ms=float(np.median(acc[mode]["mud"])) if acc[mode]["mud"] else None,
                   kin_lead_ms=float(np.median(acc[mode]["kin"])) if acc[mode]["kin"] else None,
                   det=float(np.mean(acc[mode]["det"])) if acc[mode]["det"] else None,
                   n_pre_command_alarms=int(acc[mode]["pre"]),
                   p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None,
                   adv=[float(x) for x in a])
        rows.append(row)
        lab = {"shipped": "A shipped (anchored t_on)",
               "rolling": "B rolling causal (no t_on)"}.get(name,
                                                            "C anchor late %+.0f ms" % (1000 * shift))
        f = lambda v, p="%+.2f": (p % v) if v is not None else "--"           # noqa: E731
        print("%-27s %4d %11s %8s %3d/%-2d %9s %9s %5.0f%% %5d"
              % (lab, row["n"], f(row["adv_median"]), f(row["worst"], "%+.1f"),
                 row["n_pos"], row["n"], f(row["muD_lead_ms"]), f(row["kin_lead_ms"]),
                 100 * (row["det"] or 0), row["n_pre_command_alarms"]))

    ctrl, roll = rows[0]["adv_median"], rows[1]["adv_median"]
    print("\nCONTROL: rule A must reproduce the shipped +17.0 ms -> %s"
          % ("OK" if ctrl is not None and abs(ctrl - 17.0) < 1e-6 else "*** MISMATCH ***"))
    if ctrl is not None and roll is not None:
        d = roll - ctrl
        print("Rule B uses NO knowledge of t_on and moves the advantage by %+.2f ms." % d)
        print("VERDICT: the t_on anchoring is %s"
              % ("BOOKKEEPING -- a trailing self-calibration reproduces the result"
                 if abs(d) <= 2.0 else "LOAD-BEARING -- the result depends on knowing t_on"))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(rows, open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
