"""experiments/phase_cue_detector.py -- derivative and phase-displacement muD statistics on slow commands.

slow_cue_detector.py races six muD variants at 1, 4 and 15 s command periods, and none recovers the
long-period cue. All six read the same channel, stat_matched_phase, a trailing mean of |dphi/dt|^2;
they vary the dwell and add an accumulator.

The slow-time phase is

    phi(t) = (4*pi/lambda) * r * sin(delta(t))

so the phase excursion is set by the fin angle and is the same whether the fin reaches it in 0.1 s
or 15 s. Differentiation divides by the duration: |dphi/dt| falls as 1/T and the default statistic,
its square, as 1/T^2, a factor of 2.25e4 between a 0.1 s and a 15 s command. A statistic on the
phase itself keeps the full excursion.

The cue, the causal rule, the cruise-maximum zero-false-alarm threshold, the 3-of-3 alarm and the
kinematic comparator are held fixed. Only the muD statistic changes:

    dphi-D2 max         trailing mean |dphi/dt|^2, D = 2 ms          (default arm, control)
    dphi-D100 cusum     best slow variant from slow_cue_detector     (control)
    phase-dev max       |phi - phi_cruise|, trailing mean over D = 2 ms
    phase-dev D100      |phi - phi_cruise|, trailing mean over D = 100 ms
    phase-dev cusum     CUSUM on |phi - phi_cruise|

Unwrapped phase of pure noise is a random walk, so its excursion from a baseline grows without bound
and a phase-displacement statistic can alarm on noise alone. Every row therefore reports fa from the
no-cue null (fin held still, identical noise) on the identical statistic; a phase variant detects the
weave only if its null alarm rate is low. attribution_cell then delays the fin history by 40 ms and
measures how far the alarm moves.

Reading absolute phase over seconds assumes carrier-phase coherence over the same interval. Eq. (2)
of the paper takes the bulk term as removed, and body Doppler is ambiguous at this PRF, so the
phase-displacement arms are more idealised than the derivative arm.

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

# (tag, dwell_s, channel, use_cusum). channel "dphi" is the default derivative statistic;
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

    The baseline is the mean over the leading quarter of the record, which is pre-command by
    construction, and the smoothing is a trailing mean, so no sample after the decision sample
    enters the statistic.
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
                # kinematic arm in its default configuration
                rk = np.random.default_rng(4000 + r + 991)
                azr = np.abs(np.interp(t, tf, az) + rk.normal(0, SIGMA, len(t)))
                zk = ml.stat_cusum(azr, BASE_DWELL)
                lk = causal_lead(t, zk, thr_from_cruise(t, zk, t_on), t_on)
                dets.append(1.0 if lm is not None else 0.0)
                # No-cue null on the identical statistic. Unwrapped noise phase is a random walk and
                # drifts from any fixed baseline, so this null is the key control for the
                # phase-displacement arms.
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
    """Median alarm time with the fin history delayed by `shift` seconds.

    A clean null shows the arm needs the fin to alarm; this cell tests whether the fin also sets the
    alarm time. Delaying the fin history by a fixed shift, with everything else unchanged, moves a
    fin-timed alarm by that shift. Also returns the median fraction of the fin's excursion inside the
    threshold window (t_on - 0.60 s to t_on - 0.12 s). Where the fin moves during the window that
    sets the cruise maximum, signal and threshold rise together and the crossing time no longer marks
    onset.
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

    print("Phase-channel race: derivative and phase-displacement muD statistics across command periods")
    print("  kinematic arm fixed; only the muD statistic varies.")
    print("  FA is the alarm rate of the same statistic on the no-cue null. A phase arm with a high")
    print("  FA is responding to the phase random walk.\n")
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

    print("\nA variant qualifies when it detects (det high) with a clean null (FA low) and a")
    print("positive advantage. The null runs the same statistic with the fin held still, so FA")
    print("measures alarms that do not depend on the fin.")

    # ---- attribution: shift the fin history and measure the alarm shift -------------------------
    print("\n\nAttribution: the fin history is delayed by 40 ms; a fin-timed alarm moves with it.\n")
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
    print("threshold window, the cruise maximum is itself fin-driven, so signal and threshold rise")
    print("together and the window sets the crossing time.")
    print("The same holds for any statistic whose threshold is calibrated on a window the fin")
    print("already moves in.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dict(snr_db=SNR, sigma=SIGMA, seeds=SEEDS, reps=REPS,
                           amp_factor=AMP, periods=list(PERIODS), rows=rows,
                           attribution=attr), f, indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
