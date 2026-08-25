"""experiments/online_fa_rate.py -- the false-alarm RATE of the online rule, measured.

WHY THIS EXISTS. Four independent audits converged on the same charge, and it is the strongest one
made against this letter:

    "zero false alarms" is true BY CONSTRUCTION, not by measurement. eta is the maximum of the
    statistic over a pre-command window W, so nothing inside W can exceed it. The claim is
    unfalsifiable as stated, and it cannot be compared with literature reporting P_fa = 1% or 5%.

That charge is correct for the ANCHORED rule and the letter says so. It is NOT correct for the
online rule introduced in causal_threshold_test.py, and the difference is the whole point:

    anchored   eta = max over a FIXED window W -> no sample inside W can exceed it. Vacuous.
    online     eta(t) = max over the TRAILING window [t-0.60, t-0.12] -> the current sample is NOT
               in the window that set the threshold, so it CAN exceed it. A false alarm is possible
               at every decision sample, and its rate is an empirical quantity.

So the online rule converts "zero by construction" into a number that can be measured and can be
wrong. This script measures it, on pure quiescent flight, for both arms:

  muD arm    the NO-CUE null: identical noise, identical rendering, fin held still (cue_on=False).
             This is the same null the letter already runs at every cell, now scored under the
             online threshold instead of the anchored one.
  CUSUM arm  |true a_z + N(0,sigma)| with a_z == 0, i.e. quiescent cruise at the same sigma.

An alarm is the same 3-of-3 run on the same 1 ms decision grid used everywhere else. Alarm EVENTS
are counted, not samples: after a crossing the scan skips forward until the statistic falls back
below threshold, so one excursion is one false alarm.

REPORTED: alarms per second of quiescent observation, and the mean time between false alarms. This
is a rate an operator could design to, and it is comparable -- as a rate -- with the P_fa figures
in the cited literature, which the anchored rule's "0%" is not.

    python experiments/online_fa_rate.py --json runs/ml/online_fa_rate.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
import experiments.multiclass_lead as ml                                      # noqa: E402
from experiments.dphi_sweep import return_from_fin, stat_matched_phase        # noqa: E402
from experiments.causal_threshold_test import rolling_thr, CRUISE, GUARD      # noqa: E402
from experiments.causal_dwell_test import DT_R, DECISION_GRID_S               # noqa: E402

SNR, DWELL, SIGMA = 40.0, 0.002, 0.3
DUR_S = 60.0                 # quiescent seconds per record
N_REC = 30                   # records (independent noise realisations)
NEED = 3                     # decision samples, as everywhere else


def count_alarm_events(t, stat, thr):
    """Distinct alarm EVENTS on the 1 ms decision grid, with dead time to the end of each excursion.

    Returns (n_events, observed_seconds). Samples where the threshold is not yet defined (the
    rolling buffer is still filling) are excluded from BOTH the numerator and the denominator --
    counting unobservable time as quiescent would deflate the rate.
    """
    step = max(1, int(round(DECISION_GRID_S / DT_R)))
    tt, ss, hh = t[::step], np.asarray(stat)[::step], np.asarray(thr)[::step]
    ok = np.isfinite(ss) & np.isfinite(hh)
    tt, ss, hh = tt[ok], ss[ok], hh[ok]
    if tt.size < NEED + 1:
        return 0, 0.0
    above = (ss > hh).astype(np.int8)
    # One event per MAXIMAL above-threshold run of length >= NEED. This is exactly the scan-and-
    # skip-to-end-of-excursion rule, vectorised: the loop form was O(n) in Python and the bisection
    # in matched_fa_race.py calls this ~200 times over 1.4 M decision samples, which is hours.
    d = np.diff(np.concatenate(([0], above, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    n_ev = int(((ends - starts) >= NEED).sum())
    return n_ev, float(tt[-1] - tt[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=float, default=DUR_S)
    ap.add_argument("--records", type=int, default=N_REC)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("ONLINE FALSE-ALARM RATE -- the trailing-window rule, on quiescent flight only")
    print("  %d records x %.0f s = %.0f s of no-manoeuvre observation per arm"
          % (args.records, args.dur, args.records * args.dur))
    print("  threshold: eta(t) = max over [t-%.2f, t-%.2f] s; alarm = %d of %d on a %.0f ms grid"
          % (CRUISE, GUARD, NEED, NEED, 1000 * DECISION_GRID_S))
    print("  NOTE: under this rule the current sample is NOT in the calibrating window, so a false"
          "\n        alarm is possible at every decision sample. The anchored rule forbids it.\n")

    t_fin = np.arange(0.0, args.dur, 1e-4)
    delta_still = np.zeros_like(t_fin)                     # quiescent: no fin motion at all
    rows = []

    for arm in ("muD (no-cue null)", "CUSUM (quiescent)"):
        ev_tot, sec_tot = 0, 0.0
        for k in range(args.records):
            if arm.startswith("muD"):
                # identical rendering to the measurement, fin held still
                t, s = return_from_fin(t_fin, delta_still, SNR, 7000 + k, ml.FIN_ARM_M,
                                       cue_on=False)
                stat = stat_matched_phase(t, s, DWELL)
            else:
                t = np.arange(t_fin[0], t_fin[-1], DT_R)
                rk = np.random.default_rng(7000 + k + 991)
                stat = ml.stat_cusum(np.abs(rk.normal(0, SIGMA, len(t))), DWELL)
            thr = rolling_thr(t, stat)
            ev, sec = count_alarm_events(t, stat, thr)
            ev_tot += ev
            sec_tot += sec
        rate = ev_tot / sec_tot if sec_tot else float("nan")
        mtbfa = (sec_tot / ev_tot) if ev_tot else float("inf")
        rows.append(dict(arm=arm, events=ev_tot, seconds=sec_tot, rate_per_s=rate,
                         mean_time_between_fa_s=mtbfa))
        print("  %-20s %4d alarms in %7.1f s  ->  %.4f /s   (mean time between FA: %s)"
              % (arm, ev_tot, sec_tot, rate,
                 "%.1f s" % mtbfa if np.isfinite(mtbfa) else "no false alarm observed"))

    print("\nINTERPRETATION. This is a measured rate on a rule with no knowledge of t_on, not a")
    print("count inside a window that forbids exceedance. If it is small, the letter's operating")
    print("point is defensible as an operating point; if it is large, the zero-false-alarm framing")
    print("survives only for the anchored rule and the online claim has to be withdrawn.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(snr_db=SNR, sigma=SIGMA, dwell_s=DWELL, need=NEED,
                       grid_s=DECISION_GRID_S, cruise_s=CRUISE, guard_s=GUARD,
                       dur_s=args.dur, records=args.records, arms=rows),
                  open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
