"""experiments/phase_cue_detector.py -- does the slow command fail the CUE or the CHANNEL?

THE GAP THIS FILLS. slow_cue_detector.py races six muD variants at 1, 4 and 15 s command periods
and none recovers the long-period cue. Every one of those variants reads the same channel:
stat_matched_phase, a trailing mean of |dphi/dt|^2. They vary the dwell and add an accumulator,
never the quantity being accumulated.

That matters because of how the cue scales. The slow-time phase is

    phi(t) = (4*pi/lambda) * r * sin(delta(t))

so the phase EXCURSION is set by the fin ANGLE and is the same whether the fin gets there in 0.1 s
or 15 s. Differentiating divides by the duration: |dphi/dt| falls as 1/T and the shipped statistic,
its square, falls as 1/T^2 -- a factor of 2.25e4 between a 0.1 s and a 15 s command. The excursion
that produced it never moved. So the long-period failure may be a property of the DERIVATIVE
channel rather than of the cue.

WHAT THIS RACES. The same cue, the same causal rule, the same cruise-maximum zero-false-alarm
threshold, the same 3-of-3 alarm and the same untouched kinematic comparator. Only the muD
statistic changes, and the new ones read the phase itself:

    dphi-D2 max         trailing mean |dphi/dt|^2, D = 2 ms          (the shipped arm, control)
    dphi-D100 cusum     the best slow variant from slow_cue_detector (control)
    phase-dev max       |phi - phi_cruise|, trailing mean over D
    phase-dev cusum     CUSUM on |phi - phi_cruise|

THE CONTROL THAT DECIDES IT. Unwrapped phase of pure noise is a RANDOM WALK: its excursion from a
baseline grows without bound, so a phase-displacement statistic can alarm on noise alone. The
no-cue null (fin held still, identical noise) is therefore not a formality here -- it is the whole
question. A phase variant that "detects" the 15 s weave while its null alarms at the same rate has
detected nothing. Every row below reports fa from that null on the identical statistic.

DISCLOSURE THIS FORCES IF IT WORKS. Reading absolute phase over seconds assumes carrier-phase
coherence over the same interval. Eq. (2) takes the bulk term as removed and the letter states that
body Doppler is ambiguous at this PRF, so a phase-displacement arm is more idealised than the
derivative arm it replaces. That is a real difference in what the two arms assume and it belongs in
the manuscript, not in this docstring alone.

    python experiments/phase_cue_detector.py --json runs/ml/phase_cue.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AMP, SNR, REPS, SIGMA, SEEDS = 0.2798, 40.0, 12, 0.3, 30
BASE_DWELL = 0.002
PERIODS = (0.1, 1.0, 4.0, 15.0)

# (tag, dwell_s, channel, use_cusum). channel "dphi" is the shipped derivative statistic;
# "phase" is the excursion of the unwrapped phase from its own pre-command level.
VARIANTS = (
    ("dphi-D2 max",      0.002, "dphi",  False),
    ("dphi-D100 cusum",  0.100, "dphi",  True),
    ("phase-dev max",    0.002, "phase", False),
    ("phase-dev D100",   0.100, "phase", False),
    ("phase-dev cusum",  0.002, "phase", True),
)


def _phase_excursion(t, s, dwell_s):
    """|unwrapped phase - its own leading-quarter level|, smoothed over the dwell.

    The baseline is taken from the leading quarter of the record, which is pre-command by
    construction here, so the statistic is causal in the same sense the others are: nothing after
    the decision sample enters it.
    """
    import numpy as np
    ph = np.unwrap(np.angle(s))
    m = max(8, int(0.25 * len(ph)))
    z = np.abs(ph - float(np.mean(ph[:m])))
    n = max(2, int(dwell_s / (t[1] - t[0])))
    return np.convolve(z, np.ones(n) / n, mode="full")[:len(t)]


def run_cell(arg):
    period, tag, dwell, channel, use_cusum = arg
    import numpy as np
    from scipy.stats import wilcoxon
    import experiments.multiclass_lead as ml
    from experiments.multiclass_lead import class_windows
    from experiments.dphi_sweep import return_from_fin, stat_matched_phase
    from experiments.causal_dwell_test import causal_lead, thr_from_cruise

    ml.SHAPES = dict(ml.SHAPES)
    ml.SHAPES["supersonic_cruise"] = ("sinusoid", float(period))

    def stat(t, s):
        z = (stat_matched_phase(t, s, dwell) if channel == "dphi"
             else _phase_excursion(t, s, dwell))
        return ml.stat_cusum(np.asarray(z, float), dwell) if use_cusum else z

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
                t, s = return_from_fin(tf, delta, SNR, 4000 + r, ml.FIN_ARM_M)
                lm = causal_lead(t, stat(t, s), thr_from_cruise(t, stat(t, s), t_on), t_on)
                # kinematic arm UNCHANGED, so any gain is a gain and not a moved goalpost
                rk = np.random.default_rng(4000 + r + 991)
                azr = np.abs(np.interp(t, tf, az) + rk.normal(0, SIGMA, len(t)))
                zk = ml.stat_cusum(azr, BASE_DWELL)
                lk = causal_lead(t, zk, thr_from_cruise(t, zk, t_on), t_on)
                dets.append(1.0 if lm is not None else 0.0)
                # NO-CUE NULL on the identical statistic. For a phase-displacement arm this is the
                # measurement that decides the whole question, because unwrapped noise phase is a
                # random walk and will leave any fixed baseline given enough time.
                tn, sn = return_from_fin(tf, delta, SNR, 4000 + r, ml.FIN_ARM_M, cue_on=False)
                zn = stat(tn, sn)
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
    return dict(period_s=float(period), variant=tag, dwell_s=dwell, channel=channel,
                cusum=bool(use_cusum), n=int(a.size), adv_median=med(a),
                worst=float(a.min()) if a.size else None, n_pos=int((a > 0).sum()),
                muD_lead_ms=med(mus), cusum_lead_ms=med(kins),
                det=float(np.mean(dets)) if dets else None,
                fa=float(np.mean(fas)) if fas else None,
                p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None)


def attribution_cell(arg):
    """Does the alarm move when the fin history moves?

    Detection with a clean null shows the arm NEEDS the fin to alarm. It does not show the fin
    TIMES the alarm. Delaying the fin history by a fixed shift and changing nothing else must move
    the alarm by that shift. Reported alongside how much of the fin's excursion already falls
    inside the threshold window: where the fin is in motion during the window that sets the cruise
    maximum, signal and threshold rise together and the crossing time stops being an onset.
    """
    period, tag, dwell, channel, use_cusum, shift = arg
    import numpy as np
    import experiments.multiclass_lead as ml
    from experiments.multiclass_lead import class_windows
    from experiments.dphi_sweep import return_from_fin, stat_matched_phase
    from experiments.causal_dwell_test import causal_lead, thr_from_cruise

    ml.SHAPES = dict(ml.SHAPES)
    ml.SHAPES["supersonic_cruise"] = ("sinusoid", float(period))

    alarms, fin_frac = [], []
    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        if not wins:
            continue
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
            fl = ml.drive_airframe(w2["t"], w2["a_cmd"], w2["V"], w2["alt"])
            t_on, _ = ml.onset_from_achieved(fl["t"], fl["az"])
            if t_on is None:
                continue
            tf, dd = fl["t"], np.abs(fl["delta"])
            m = (tf >= t_on - 0.60) & (tf <= t_on - 0.12)
            full = float(dd.max() - dd.min()) or 1.0
            fin_frac.append(float(dd[m].max() - dd[m].min()) / full)
            for r in range(REPS):
                t, s = return_from_fin(tf, fl["delta"], SNR, 4000 + r,
                                       ml.FIN_ARM_M, shift_s=shift)
                z = (stat_matched_phase(t, s, dwell) if channel == "dphi"
                     else _phase_excursion(t, s, dwell))
                if use_cusum:
                    z = ml.stat_cusum(np.asarray(z, float), dwell)
                lead = causal_lead(t, z, thr_from_cruise(t, z, t_on), t_on)
                if lead is not None:
                    alarms.append(1000.0 * (t_on - lead))
    return dict(period_s=float(period), variant=tag, shift_ms=1000.0 * shift,
                n=len(alarms),
                alarm_ms=float(np.median(alarms)) if alarms else None,
                fin_travel_in_window=float(np.median(fin_frac)) if fin_frac else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--procs", type=int, default=max(1, min(10, cpu_count() - 2)))
    args = ap.parse_args()

    print("PHASE-CHANNEL RACE -- is the long-period failure the CUE's or the DERIVATIVE's?")
    print("  kinematic arm untouched; only the muD statistic varies.")
    print("  FA is the no-cue null on the SAME statistic. A phase arm with a high null has")
    print("  detected its own random walk, not the fin.\n")
    jobs = [(p, t, d, ch, c) for p in PERIODS for (t, d, ch, c) in VARIANTS]
    with Pool(args.procs) as pool:
        rows = pool.map(run_cell, jobs)

    print("%-7s %-18s %5s %10s %8s %7s %7s %10s"
          % ("period", "variant", "n", "adv ms", "det", "FA", "n_pos", "p"))
    print("-" * 84)
    for r in rows:
        print("%-7s %-18s %5d %10s %7s%% %6s%% %7s %10s"
              % ("%.1f s" % r["period_s"], r["variant"], r["n"],
                 "%+.2f" % r["adv_median"] if r["adv_median"] is not None else "--",
                 "%.0f" % (100 * r["det"]) if r["det"] is not None else "--",
                 "%.0f" % (100 * r["fa"]) if r["fa"] is not None else "--",
                 "%d/%d" % (r["n_pos"], r["n"]),
                 "%.1e" % r["p"] if r["p"] is not None else "--"))

    print("\nA variant COUNTS only if it converts (det high) with a clean null (FA low) AND a")
    print("positive advantage. Detection alone is not evidence: the null measures what the same")
    print("statistic does when the fin never moves.")

    # ---- ATTRIBUTION. A clean null is necessary and not sufficient -----------------------------
    print("\n\nATTRIBUTION -- delay the fin history and the alarm must follow it.\n")
    ATTR = (("dphi-D2 max", 0.002, "dphi", False),
            ("phase-dev max", 0.002, "phase", False))
    jobs = [(p, t, d, ch, c, sh) for p in PERIODS for (t, d, ch, c) in ATTR
            for sh in (0.0, 0.040)]
    with Pool(args.procs) as pool:
        arows = pool.map(attribution_cell, jobs)

    attr = []
    print("%-8s %-16s %7s %11s %11s %10s   %s"
          % ("period", "variant", "n", "alarm@0", "alarm@40ms", "moved", "fin travel in W"))
    print("-" * 92)
    for p in PERIODS:
        for tag, _d, _ch, _c in ATTR:
            a0 = next((r for r in arows if r["period_s"] == p and r["variant"] == tag
                       and r["shift_ms"] == 0.0), None)
            a1 = next((r for r in arows if r["period_s"] == p and r["variant"] == tag
                       and r["shift_ms"] == 40.0), None)
            moved = (a1["alarm_ms"] - a0["alarm_ms"]) if (a0 and a1 and a0["alarm_ms"]
                                                          is not None and a1["alarm_ms"]
                                                          is not None) else None
            attr.append(dict(period_s=p, variant=tag, n=a0["n"] if a0 else 0,
                             alarm_ms=a0["alarm_ms"] if a0 else None,
                             alarm_shifted_ms=a1["alarm_ms"] if a1 else None,
                             moved_ms=moved,
                             fin_travel_in_window=a0["fin_travel_in_window"] if a0 else None,
                             attributable=bool(moved is not None and moved >= 30.0)))
            print("%-8s %-16s %7s %11s %11s %10s   %s"
                  % ("%.1f s" % p, tag, a0["n"] if a0 else 0,
                     "%.1f" % a0["alarm_ms"] if a0 and a0["alarm_ms"] is not None else "--",
                     "%.1f" % a1["alarm_ms"] if a1 and a1["alarm_ms"] is not None else "--",
                     "%+.1f ms" % moved if moved is not None else "--",
                     "%.0f%%" % (100 * a0["fin_travel_in_window"])
                     if a0 and a0["fin_travel_in_window"] is not None else "--"))

    print("\nA fin-timed alarm moves by ~+40 ms. Where the fin is already in motion inside the")
    print("threshold window, the cruise maximum is itself fin-driven, signal and threshold rise")
    print("together, and the crossing time is set by the window rather than by the maneuver.")
    print("That bound is a property of onset detection against a pre-command baseline, not of")
    print("either statistic.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dict(snr_db=SNR, sigma=SIGMA, seeds=SEEDS, reps=REPS,
                           amp_factor=AMP, periods=list(PERIODS), rows=rows,
                           attribution=attr), f, indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
