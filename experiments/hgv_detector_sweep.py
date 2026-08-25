"""experiments/hgv_detector_sweep.py -- was HGV undetectable, or was the detector wrong?

THE OBJECTION THAT PROMPTED THIS. multiclass_lead.py reports 0 % micro-Doppler detection on HGV and
a -246 to -292 ms loss against CUSUM/GLR, and that was written up as a property of the class. It is
not obviously that. HGV in this testbed flies a skip-glide plus lateral S-weave -- 6-11 skips and
4-7 weave cycles over a 300-480 s flight -- so its maneuver period is TENS OF SECONDS. It was tested
with a detector matched to a short broadband transient: a 2 ms trailing mean of |dphi/dt|^2, chosen
because it was right for a step-commanded cruise missile whose whole budget is ~35 ms.

A 2 ms detector on a 36 s maneuver is a mismatched filter, and "mismatched filter finds nothing" is
a statement about the filter. The class was excluded on that basis and the exclusion has to be
earned instead: sweep the dwell over four decades and try statistics built for slow and PERIODIC
modulation, not for transients.

WHAT IS SWEPT
  dwell            2 ms -> 5 s, four decades, on the same fin history
  matched          trailing-mean |dphi/dt|^2 (the incumbent; expected to fail at long dwell too,
                   because averaging a zero-mean oscillation cancels it)
  band-power       power of |dphi/dt| in a band placed on the WEAVE frequency, not on the 3-10 Hz
                   band inherited from the cruise-missile study -- a weave at 4-7 cycles over
                   300-480 s lives at ~0.01-0.02 Hz, three orders of magnitude below where the
                   incumbent band looks
  cadence          peak of the periodogram of |dphi/dt| over the dwell: a periodicity detector,
                   which is what a quasi-periodic maneuver actually calls for

Every arm keeps the protocol that survived review: rendered from delta(t) (observable, never the
command), causal, cruise-maximum threshold (zero false alarms before the event), and a NO-CUE NULL
at every cell so a detection rate cannot be confused with an alarm rate.

    python experiments/hgv_detector_sweep.py
    python experiments/hgv_detector_sweep.py --seeds 6 --json runs/ml/hgv_sweep.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from experiments.class_profiles import (                                      # noqa: E402
    load_class, lateral_signal, decompose, describe, feasibility, ONSET_G_MS2,
)
from experiments.multiclass_lead import (                                     # noqa: E402
    drive_airframe, onset_from_achieved, synth_command, DT_FINE, PRE_S, FIN_ARM_M,
)
from experiments.causal_dwell_test import causal_lead, thr_from_cruise, DT_R  # noqa: E402
from experiments.dphi_sweep import return_from_fin, dphi                      # noqa: E402

DWELLS = (0.002, 0.010, 0.050, 0.200, 1.000, 5.000)


def stat_matched(t, s, dwell):
    """The incumbent: trailing-mean |dphi/dt|^2."""
    n = max(2, int(dwell / DT_R))
    return np.convolve(dphi(s) ** 2, np.ones(n) / n, mode="full")[:len(t)]


def stat_band(t, s, dwell, f_lo, f_hi):
    """Band power of |dphi/dt| over a trailing window, in a band placed on the WEAVE rate.

    The 3-10 Hz band the cruise-missile study uses is meaningless here: an HGV weave at 4-7 cycles
    across 300-480 s sits near 0.01-0.02 Hz. The band has to follow the target, not the previous
    experiment.
    """
    n = max(4, int(dwell / DT_R))
    f = np.fft.rfftfreq(n, d=DT_R)
    band = (f >= f_lo) & (f <= f_hi)
    if not band.any():
        return None
    x = dphi(s)
    out = np.full(len(t), np.nan)
    step = max(1, n // 8)                       # stride: a 5 s dwell at 2 kHz is 10k-point FFTs
    idx = range(n, len(t), step)
    vals = []
    for i in idx:
        w = x[i - n:i]
        w = w - w.mean()
        vals.append((np.abs(np.fft.rfft(w)) ** 2)[band].sum() / n)
    out[list(idx)] = vals
    return _ffill(out)


def stat_cadence(t, s, dwell):
    """Peak periodogram power over the dwell -- a PERIODICITY detector.

    A quasi-periodic maneuver is exactly what a cadence estimator is for, and nothing in this
    project has tried one: every statistic so far has been an energy detector, which is matched to
    a transient and blind to structure.
    """
    n = max(8, int(dwell / DT_R))
    x = dphi(s)
    out = np.full(len(t), np.nan)
    step = max(1, n // 8)
    idx = range(n, len(t), step)
    vals = []
    for i in idx:
        w = x[i - n:i]
        w = w - w.mean()
        P = np.abs(np.fft.rfft(w)) ** 2
        vals.append(float(P[1:].max()) if len(P) > 1 else 0.0)
    out[list(idx)] = vals
    return _ffill(out)


def _ffill(a):
    idx = np.where(np.isfinite(a))[0]
    if not len(idx):
        return a
    first = idx[0]
    out = a.copy()
    for i in range(first + 1, len(out)):
        if not np.isfinite(out[i]):
            out[i] = out[i - 1]
    return out


def hgv_event(rng):
    """One HGV trajectory through the dataset path, and its first feasible reversal as a window."""
    rec = load_class("hgv", rng=rng, max_attempts=40)
    d = describe(rec)
    sig, _src = lateral_signal(rec)
    man, _trim, _cut = decompose(sig)
    t = rec["t"]
    alt_all, amax_all = feasibility(rec)
    above = man > ONSET_G_MS2
    rises = np.where(np.diff(above.astype(int)) == 1)[0] + 1
    falls = np.where(np.diff(above.astype(int)) == -1)[0] + 1
    for i_on in rises:
        amax = amax_all[i_on]
        nxt = falls[falls > i_on]
        i_end = int(nxt[0]) if len(nxt) else len(man)
        amp = float(man[i_on:i_end].max()) if i_end > i_on else float(man[i_on])
        if np.isfinite(amax) and amax > 0 and amp > amax:
            continue
        # weave period from the reversal spacing, so the band follows the target
        per = float(np.median(np.diff(t[rises]))) * 2.0 if len(rises) > 2 else 40.0
        alt = float(alt_all[i_on])
        V = float(np.linalg.norm(rec["V"][i_on]))
        tt, aa, t_cmd = synth_command("sinusoid", per, amp, PRE_S, 0.4)
        return dict(t=tt, a_cmd=aa, alt=alt, V=V, t_cmd=t_cmd, period=per,
                    amp_g=amp / 9.80665, n_rev=len(rises))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("HGV DETECTOR SWEEP -- was the class undetectable, or the detector mismatched?\n")
    print("  Every arm: rendered from delta(t), causal, cruise-maximum threshold, no-cue NULL.")
    print("  A cell only counts if detection beats its own null.\n")

    wins = []
    for sd in range(args.seeds):
        try:
            w = hgv_event(np.random.default_rng(51000 + sd))
        except Exception as e:                                       # noqa: BLE001
            print("  seed %d: %s" % (sd, str(e)[:60]))
            continue
        if w:
            wins.append(w)
    if not wins:
        print("  no HGV trajectory passed the Tier-0 gate with a feasible reversal.")
        return
    per = float(np.median([w["period"] for w in wins]))
    amp = float(np.median([w["amp_g"] for w in wins]))
    print("  %d trajectories | median weave period %.1f s (%.4f Hz) | reversal amplitude %.2f g\n"
          % (len(wins), per, 1.0 / per, amp))

    f0 = 1.0 / per
    arms = [("matched", lambda t, s, d: stat_matched(t, s, d)),
            ("band@weave", lambda t, s, d: stat_band(t, s, d, 0.2 * f0, 5.0 * f0)),
            ("band 3-10Hz", lambda t, s, d: stat_band(t, s, d, 3.0, 10.0)),
            ("cadence", lambda t, s, d: stat_cadence(t, s, d))]

    print("  %-14s %9s %7s %7s %11s" % ("statistic", "dwell", "det", "FA", "median lead"))
    print("  " + "-" * 56)
    out = []
    for name, fn in arms:
        for dw in DWELLS:
            leads, det, fa, formed = [], 0, 0, 0
            for w in wins:
                fl = drive_airframe(w["t"], w["a_cmd"], w["V"], w["alt"])
                t_on, _i = onset_from_achieved(fl["t"], fl["az"])
                if t_on is None:
                    continue
                for r in range(args.reps):
                    t, s = return_from_fin(fl["t"], fl["delta"], args.snr, 4000 + r, FIN_ARM_M)
                    st = fn(t, s, dw)
                    if st is None:
                        continue
                    formed += 1
                    lk = causal_lead(t, st, thr_from_cruise(t, st, t_on), t_on)
                    if lk is not None:
                        det += 1; leads.append(1000 * lk)
                    tn, sn = return_from_fin(fl["t"], fl["delta"], args.snr, 4000 + r,
                                             FIN_ARM_M, cue_on=False)
                    stn = fn(tn, sn, dw)
                    if stn is not None and causal_lead(
                            tn, stn, thr_from_cruise(tn, stn, t_on), t_on) is not None:
                        fa += 1
            if formed == 0:
                print("  %-14s %8.0fms %7s %7s %11s" % (name, 1000 * dw, "--", "--", "unformable"))
                out.append(dict(stat=name, dwell=dw, unformable=True))
                continue
            print("  %-14s %8.0fms %6.0f%% %6.0f%% %11s"
                  % (name, 1000 * dw, 100 * det / formed, 100 * fa / formed,
                     ("%+.1f" % np.median(leads)) if leads else "--"))
            out.append(dict(stat=name, dwell=dw, det=det / formed, fa=fa / formed,
                            lead=float(np.median(leads)) if leads else None))

    good = [o for o in out if not o.get("unformable") and o.get("lead") is not None
            and o["det"] >= 0.5 and o["fa"] <= 0.05 and o["lead"] > 0]
    print("\n  CELLS THAT CONVERT (det >= 50%%, no-cue FA <= 5%%, positive lead): %d of %d"
          % (len(good), len([o for o in out if not o.get("unformable")])))
    for o in sorted(good, key=lambda x: -x["lead"])[:6]:
        print("    %-14s %5.0f ms dwell -> %+.1f ms at %.0f%% detection, %.0f%% FA"
              % (o["stat"], 1000 * o["dwell"], o["lead"], 100 * o["det"], 100 * o["fa"]))
    if not good:
        print("    none. The exclusion is now EARNED rather than assumed: four statistic families")
        print("    across four decades of dwell, each with its own null, and nothing converts.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print("\n  wrote %s" % args.json)


if __name__ == "__main__":
    main()
