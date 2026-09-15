"""experiments/online_fa_rate.py -- false-alarm rate of the online threshold rule on quiescent flight.

The anchored threshold, eta = max of the statistic over a fixed pre-command window W, cannot be
exceeded by any sample inside W, so its false-alarm count there is zero by definition. The online
rule of causal_threshold_test.py sets eta(t) = max over the trailing window [t-0.60, t-0.12]. The
current sample lies outside the window that sets its threshold, so a false alarm can occur at any
decision sample. This script measures that rate on quiescent flight for both arms:

  muD arm    the no-cue null: the same noise and rendering with the fin held still (cue_on=False),
             scored under the online threshold.
  CUSUM arm  |a_z + N(0, sigma)| with a_z = 0, quiescent cruise at the same sigma.

An alarm is a 3-of-3 run on the 1 ms decision grid used throughout. Events are counted, one per
above-threshold excursion of at least three decision samples. The output is alarms per second of
quiescent observation and the mean time between false alarms.

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
    """Distinct alarm events on the 1 ms decision grid, with dead time to the end of each excursion.

    Returns (n_events, observed_seconds). Samples where the statistic or threshold is undefined,
    including the interval while the rolling buffer fills, are excluded from both the event count
    and the observed time.
    """
    step = max(1, int(round(DECISION_GRID_S / DT_R)))
    tt, ss, hh = t[::step], np.asarray(stat)[::step], np.asarray(thr)[::step]
    ok = np.isfinite(ss) & np.isfinite(hh)
    tt, ss, hh = tt[ok], ss[ok], hh[ok]
    if tt.size < NEED + 1:
        return 0, 0.0
    above = (ss > hh).astype(np.int8)
    # One event per maximal above-threshold run of length >= NEED, which is the scan-and-skip-to-
    # end-of-excursion rule in vectorised form. The bisection in matched_fa_race.py calls this on
    # every calibration record at every step.
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
    print("  Under this rule the current sample lies outside its calibrating window, so a false"
          "\n  alarm can occur at any decision sample.\n")

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

    print("\nThe rate is alarm events per second of observed quiescent time, under a threshold that")
    print("uses no knowledge of t_on. Observed time runs from the first to the last decision sample")
    print("with a defined threshold. The mean time between false alarms is observed time divided")
    print("by the event count.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(snr_db=SNR, sigma=SIGMA, dwell_s=DWELL, need=NEED,
                       grid_s=DECISION_GRID_S, cruise_s=CRUISE, guard_s=GUARD,
                       dur_s=args.dur, records=args.records, arms=rows),
                  open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
