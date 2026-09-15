"""experiments/endgame_lead.py -- detection lead on commands produced by a guidance law.

The paper's duration axis is swept by imposing a command shape on a class whose generator samples
at 0.5 s, since a command shorter than one sample cannot come out of that generator.
trajectory_generators/endgame.py flies a proportional-navigation engagement at 1e-4 s, so the
command's amplitude and shape are outputs of the guidance law and the closing geometry. This script
runs the paper's detector, comparator, threshold rule and null on that command.

An engagement produces two events. The acquisition instant is an input to the engagement; the
command's amplitude and shape at acquisition come from the guidance law. The jink, the missile's
response to the target's break, has fully emergent timing.

    acquisition  the seeker closes the loop on a vehicle that has been coasting unguided, so the
                 pre-command window is quiescent. This is the endgame analogue of the paper's
                 commanded step.
    jink         the target breaks, the line-of-sight rate responds, and the command follows.

Acquisition converts at the fastest loops and the jink does not convert anywhere in the sweep, so
this guidance-sourced trajectory family shows the command-duration dependence reported in the paper.

Sweep axis. Zarchan (Science and Global Security 8(1):99-124, 1999) works his examples at
guidance-system time constants of 0.05, 0.1, 0.2 and 0.5 s and attributes the dominant part of the
endoatmospheric total to the flight control system; he gives no fielded range. sim.sixdof models
the flight control system, so the sweep varies only the seeker-filter lag tau_f and reports the
measured total from endgame.loop_time_constant(). The airframe alone measures 0.186 s, and the sweep
spans 0.186-0.479 s.

Controls, reported per cell:

    null         the fin held still, with identical noise and threshold rule.
    attribution  the fin history delayed by 40 ms with nothing else changed; an alarm timed by the
                 fin moves by 40 ms. Reported with the fraction of the fin's travel that falls
                 inside the threshold window. Below ATTR_DET_FLOOR detection the alarm median comes
                 from a few marginal realizations, and the row is flagged.
    chain        the same decision taken by thresholding the fin history directly, with no radar
                 rendering, receiver noise or statistic. It checks that no result is an artifact
                 of the signal processing. At 40 dB the processing chain is invertible with
                 respect to the decision for any fin history, so this control does not
                 distinguish a guidance-sourced cue from an imposed command.

measure() is called with two inputs replaced, the fin history and the onset rule. The comparator is
Page CUSUM on true lateral acceleration plus noise. It does not estimate the state it thresholds, so
no tracking-filter lag enters the comparison, as in the paper. Both
arms are timed against the same substituted onset. The self-check at the end of the run compares
the substituted onset rule with the project's rule where the pre-command trend is zero.

    python experiments/endgame_lead.py --json runs/ml/endgame_lead.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The paper's operating point, as in runs/ml/shape_period.json, true_shape.json and phase_cue.json:
# SNR 40 dB, sigma 0.3 m/s^2, 12 noise reps, 30 trajectories.
SNR, REPS, SIGMA, SEEDS, DWELL = 40.0, 12, 0.3, 30, 0.002
SHIFT_S = 0.040                       # the attribution delay, as in phase_cue_detector.py
TAUS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
KINDS = ("acquisition", "jink")
# Midcourse lateral activity, in g, run at the fastest loop (tau_f = 0), where the acquisition cue
# converts. 0.0 repeats the tau_f = 0 row of the main sweep.
MIDCOURSE_G = (0.0, 0.25, 0.5, 1.0, 2.0)
ATTR_DET_FLOOR = 0.25   # detection rate below which attribution and advantage are flagged as
                        # unreadable; see _summarise.


def _trend_onset(ev):
    """Return an onset function that applies the project's departure rule relative to a trend.

    multiclass_lead.onset_from_achieved thresholds |a_z| at 2.0 m/s^2. A missile answering a target
    break is still taking out its heading error, so |a_z| already exceeds the threshold when the
    window opens and that rule would return the first sample. The returned function applies the
    same 2.0 m/s^2 threshold to the residual against `ach_trend`, the linear trend of the achieved
    acceleration over the pre-command window. At acquisition that trend is zero and the two rules
    agree.
    """
    import numpy as np
    from experiments.class_profiles import ONSET_G_MS2
    from experiments.multiclass_lead import DT_FINE

    c0, c1 = ev["ach_trend"]
    i_pre = int(ev["i_pre"])

    def onset(t, az, need_s=0.005):
        t = np.asarray(t, float)
        need = max(2, int(need_s / DT_FINE))
        ab = np.abs(np.asarray(az, float) - (c0 + c1 * t)) > ONSET_G_MS2
        for i in range(i_pre, len(ab) - need):
            if ab[i:i + need].all():
                return float(t[i]), i
        return None, -1
    return onset


def run_cell(arg):
    """One (tau_f, seed, midcourse_g): fly the engagement once, measure both of its events on it."""
    tau_f, seed, mid_g = arg
    import numpy as np
    import experiments.multiclass_lead as ml
    from experiments.dphi_sweep import return_from_fin, stat_matched_phase
    from experiments.causal_dwell_test import causal_lead, thr_from_cruise, DT_R
    from trajectory_generators.endgame import engagement, events, command_timescale

    eng = engagement(rng=np.random.default_rng(90000 + seed), tau_f=tau_f, midcourse_g=mid_g)
    out = []
    real_drive, real_onset = ml.drive_airframe, ml.onset_from_achieved
    for ev in events(eng):
        ts = command_timescale(ev)
        tf, delta, az = ev["t"], ev["delta"], ev["a_ach"]

        # Two of measure()'s inputs are replaced. The flown fin history replaces the re-driven one,
        # and the onset rule is replaced by its trend-relative form, which moves t_on and with it
        # the budget and the threshold window. The rendering, statistic, threshold rule, comparator
        # and null are unchanged.
        #
        # On the acquisition event the pre-command trend is zero, so both onset rules should return
        # the same t_on; `equiv` records whether they do, and main() reports it.
        ml.drive_airframe = lambda *_a, **_k: dict(t=tf, delta=delta, az=az, cmd=ev["a_cmd"])
        ml.onset_from_achieved = _trend_onset(ev)
        # Equivalence is checked only on acquisition events with no midcourse steering. Under
        # midcourse steering the vehicle is already accelerating when the window opens, the native
        # rule fires on that, and the two rules are expected to differ.
        t_native, _iv = real_onset(tf, az)                                     # noqa: F821
        t_trend, _it = ml.onset_from_achieved(tf, az)
        equiv = (None if ev["kind"] != "acquisition" or mid_g > 0.0
                 or t_native is None or t_trend is None
                 else bool(abs(t_native - t_trend) <= 2.0 * float(tf[1] - tf[0])))
        try:
            win = dict(t=tf, a_cmd=ev["a_cmd"], V=ev["V"], alt=ev["alt"], t_cmd=ev["t_cmd"])
            m = ml.measure(win, SNR, REPS, DWELL, kin_noise=SIGMA)
            t_on, _ = ml.onset_from_achieved(tf, az)
        finally:
            ml.drive_airframe, ml.onset_from_achieved = real_drive, real_onset
        if m is None or t_on is None:
            continue

        # --- attribution, and how much fin travel the threshold window already contains ---------
        dd = np.abs(delta)
        w = (tf >= t_on - 0.60) & (tf <= t_on - 0.12)
        full = float(dd.max() - dd.min()) or 1.0
        fin_in_w = float(dd[w].max() - dd[w].min()) / full if w.any() else float("nan")
        alarms = {}
        for sh in (0.0, SHIFT_S):
            a = []
            for r in range(REPS):
                t, s = return_from_fin(tf, delta, SNR, 4000 + r, ml.FIN_ARM_M, shift_s=sh)
                z = stat_matched_phase(t, s, DWELL)
                lead = causal_lead(t, z, thr_from_cruise(t, z, t_on), t_on)
                if lead is not None:
                    a.append(1000.0 * (t_on - lead))
            alarms[sh] = float(np.median(a)) if a else None

        # --- chain removal: the same decision on the fin history itself -------------------------
        tg = np.arange(tf[0], tf[-1], DT_R)
        dg = np.interp(tg, tf, delta)
        n = max(2, int(DWELL / DT_R))
        raw = (np.abs(np.diff(dg, prepend=dg[0])) / DT_R) ** 2
        zdir = np.convolve(raw, np.ones(n) / n, mode="full")[:len(tg)]
        ldir = causal_lead(tg, zdir, thr_from_cruise(tg, zdir, t_on), t_on)

        # The reported advantage is paired against Page CUSUM (KIN_ARMS[1]), the paper's comparator.
        # measure()'s top-level `adv` uses the trailing-mean arm and is stored as adv_tmean. All
        # arms are stored.
        arms = m.get("arms") or {}
        cus = arms.get("CUSUM Page54") or {}
        out.append(dict(tau_f=float(tau_f), seed=int(seed), mid_g=float(mid_g), kind=ev["kind"],
                        onset_equiv=equiv,
                        t_rise=ts["t_rise"], d_rate=ts["d_rate"],
                        cmd_amp=ts["amp"], truncated=ts["truncated"],
                        budget_ms=m["budget_ms"], det=m["det"], fa=m["fa"],
                        muD=m["muD"], adv=cus.get("adv"), kin=cus.get("lead"),
                        adv_tmean=m.get("adv"), kin_tmean=m["kin"],
                        arms={k: dict(lead=v.get("lead"), adv=v.get("adv"), n=v.get("n"))
                              for k, v in arms.items()},
                        fin_in_w=fin_in_w, alarm_ms=alarms[0.0],
                        alarm_shift_ms=alarms[SHIFT_S],
                        direct_lead_ms=None if ldir is None else 1000.0 * ldir))
    return out


def calibrate_periods():
    """Measure the published duration axis with this experiment's rise-time clock.

    experiments/shape_period_sweep.py reports detection against the period of a commanded sinusoid,
    and endgame.command_timescale reports a rise measured from a 2 m/s^2 departure. A sinusoid's
    quarter-period starts at zero and the rise does not, with an offset that depends on amplitude,
    so period/4 is not a valid conversion. The sweep's sinusoids are run through command_timescale
    at the sweep's amplitude to give a measured map.

    Returns [(period_s, t_rise_s, d_rate)] over shape_period_sweep.PERIODS.
    """
    import numpy as np
    import experiments.multiclass_lead as ml
    from experiments.shape_period_sweep import PERIODS, AMP
    from trajectory_generators.endgame import command_timescale, _departure

    out = []
    for T in PERIODS:
        # Amplitude as flown by the sweep: the sampled command from the same class_windows call,
        # scaled by AMP.
        wins, _d = ml.class_windows("supersonic_cruise", rng=np.random.default_rng(90000),
                                    amp_factor=AMP)
        if not wins:
            continue
        amp = float(np.max(np.abs(wins[0]["a_cmd"]))) * AMP
        t, a, t_pre = ml.synth_command("sinusoid", float(T), amp, ml.PRE_S, ml.POST_S)
        fl = ml.drive_airframe(t, a, wins[0]["V"], wins[0]["alt"])
        i_pre = int(round(t_pre / (t[1] - t[0])))
        _tc, i_cmd, trend = _departure(t, a, 0, i_pre)
        ts = command_timescale(dict(t=t, a_cmd=a, delta=fl["delta"], cmd_trend=trend,
                                    i_cmd=i_cmd if i_cmd >= 0 else i_pre))
        out.append((float(T), ts["t_rise"], ts["d_rate"]))
    return out


def monotone_prefix(cal):
    """The strictly increasing prefix of the calibration, which is the invertible part.

    multiclass_lead.synth_command sizes its post-window from where the command reaches three times
    the departure threshold and caps it at 3 s, so a long-period sinusoid is cut off before it
    peaks and its measured rise stops tracking its period: rise/period is 0.241 at 0.1-1.0 s, then
    0.191, 0.091 and 0.044 as the cap takes effect, and the rise is not monotone. Rises outside the
    prefix get no equivalent period.
    """
    import numpy as np
    out = []
    for p, r, dr in cal:
        if r is None or not np.isfinite(r) or r <= 0:
            break
        if out and r <= out[-1][1]:
            break
        out.append((p, r, dr))
    return out


def period_equiv(t_rise, cal):
    """The published period whose measured rise matches `t_rise`, or None if outside the map."""
    import numpy as np
    pts = monotone_prefix(cal)
    if len(pts) < 2 or t_rise is None or not np.isfinite(t_rise) or t_rise <= 0:
        return None
    rises = [r for _p, r, _d in pts]
    if t_rise < min(rises) or t_rise > max(rises):
        return None
    return float(np.exp(np.interp(np.log(t_rise), np.log(rises),
                                  np.log([p for p, _r, _d in pts]))))


def _fmt(v, spec):
    return (spec % v) if v is not None else "--"


def _agg(rows, key):
    import numpy as np
    v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.median(v)) if v else None


def _summarise(sel, **fixed):
    """One reported row: medians over trajectories, with the advantage bootstrapped and tested."""
    import numpy as np
    from scipy.stats import wilcoxon

    a = np.array([r["adv"] for r in sel if r.get("adv") is not None], float)
    rec = dict(fixed, n_traj=len(sel), n_paired=int(a.size),
               t_rise_s=_agg(sel, "t_rise"), d_rate=_agg(sel, "d_rate"),
               cmd_amp=_agg(sel, "cmd_amp"), budget_ms=_agg(sel, "budget_ms"),
               det=float(np.mean([r["det"] for r in sel])),
               fa=float(np.mean([r["fa"] for r in sel])),
               muD_ms=_agg(sel, "muD"), kin_ms=_agg(sel, "kin"),
               adv_tmean=_agg(sel, "adv_tmean"), kin_tmean_ms=_agg(sel, "kin_tmean"),
               fin_in_w=_agg(sel, "fin_in_w"), alarm_ms=_agg(sel, "alarm_ms"),
               alarm_shift_ms=_agg(sel, "alarm_shift_ms"),
               direct_lead_ms=_agg(sel, "direct_lead_ms"),
               n_truncated=int(sum(1 for r in sel if r.get("truncated"))))
    if a.size >= 3:
        rng = np.random.default_rng(20260813)
        bt = np.median(rng.choice(a, (4000, a.size)), axis=1)
        rec.update(adv_median=float(np.median(a)), lo=float(np.percentile(bt, 2.5)),
                   hi=float(np.percentile(bt, 97.5)), n_pos=int((a > 0).sum()),
                   worst=float(a.min()),
                   p=float(wilcoxon(a).pvalue) if np.any(a != 0) else None)
    if rec["alarm_ms"] is not None and rec["alarm_shift_ms"] is not None:
        rec["moved_ms"] = rec["alarm_shift_ms"] - rec["alarm_ms"]
    # Below ATTR_DET_FLOOR the alarm median comes from a few marginal realizations, and its movement
    # under the fin delay is noise. Such rows are kept and flagged.
    rec["attr_readable"] = bool(rec["det"] >= ATTR_DET_FLOOR and rec.get("moved_ms") is not None)
    # The same floor applies to the advantage. A pair forms only where both arms alarm, so below the
    # floor the surviving pairs are selected by the outcome.
    rec["adv_readable"] = bool(rec["det"] >= ATTR_DET_FLOOR and "adv_median" in rec)
    return rec


def _print_rows(rows, axis, label):
    print("%-8s %-7s %-12s %6s %8s %8s %8s %6s %6s %9s %9s   %s"
          % (label, "T_63", "event", "n", "rise s", "= per s", "budget", "det", "FA", "muD ms",
             "CUSUM ms", "ADVANTAGE ms [95% CI] p"))
    print("-" * 136)
    for r in rows:
        adv = ("%+8.2f [%+.2f,%+.2f] p=%s" % (r["adv_median"], r["lo"], r["hi"],
                                              ("%.3g" % r["p"]) if r.get("p") else "n/a")
               if "adv_median" in r else "-- (n=%d paired)" % r["n_paired"])
        if "adv_median" in r and not r.get("adv_readable"):
            adv = "[%s  survivors only, n=%d]" % (adv.strip(), r["n_paired"])
        f = lambda k, s: (s % r[k]) if r.get(k) is not None else "--"          # noqa: E731
        print("%-8.2f %-7.3f %-12s %6d %8s %8s %8s %6.0f%% %5.0f%% %9s %9s   %s"
              % (r[axis], r["loop_T63_s"], r["kind"], r["n_traj"], f("t_rise_s", "%.3f"),
                 f("period_equiv_s", "%.2f"), f("budget_ms", "%.1f"),
                 100 * r["det"], 100 * r["fa"],
                 f("muD_ms", "%+.1f"), f("kin_ms", "%+.1f"), adv))


def _print_controls(rows, axis, label):
    print("%-8s %-12s %10s %12s %12s %12s %12s"
          % (label, "event", "fin in W", "alarm ms", "alarm+40 ms", "moved ms", "chain-off ms"))
    print("-" * 88)
    for r in rows:
        f = lambda k, s: (s % r[k]) if r.get(k) is not None else "--"          # noqa: E731
        moved = (f("moved_ms", "%+.1f") if r.get("attr_readable")
                 else ("[%s]" % f("moved_ms", "%+.1f")))
        print("%-8.2f %-12s %9s%% %12s %12s %12s %12s"
              % (r[axis], r["kind"],
                 ("%.0f" % (100 * r["fin_in_w"])) if r.get("fin_in_w") is not None else "--",
                 f("alarm_ms", "%.1f"), f("alarm_shift_ms", "%.1f"), moved,
                 f("direct_lead_ms", "%+.1f")))
    print("   [bracketed] = detection below %.0f%%; too few alarms for the movement to be read,"
          " and any advantage there is over an outcome-selected subset" % (100 * ATTR_DET_FLOOR))


def main():
    from trajectory_generators.endgame import loop_time_constant, TAU_F_BAND, T_GO_FLOOR

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--procs", type=int, default=max(1, min(10, cpu_count() - 2)))
    args = ap.parse_args()

    print("Endgame lead: the paper's detector on commands produced by proportional navigation")
    print("  guidance integrated at 1e-4 s; the command is an output of the guidance law")
    print("  SNR %.0f dB | dwell %.0f ms | sigma %.2f | %d noise reps | %d trajectories per cell"
          % (SNR, 1000 * DWELL, SIGMA, REPS, args.seeds))
    print("  advantage is paired against Page CUSUM, as in the paper")
    print("  engagement not modeled inside t_go = %.2f s\n" % T_GO_FLOOR)

    jobs = ([(t, s, 0.0) for t in TAUS for s in range(args.seeds)]
            + [(0.0, s, g) for g in MIDCOURSE_G[1:] for s in range(args.seeds)])
    with Pool(args.procs) as pool:
        got = pool.map(run_cell, jobs)
    rows = [r for chunk in got for r in chunk]
    loop = {t: loop_time_constant(t)[0] for t in TAUS}

    cal = calibrate_periods()
    print("0. Axis calibration. The published duration sweep's sinusoids, measured with this")
    print("   experiment's rise-time clock.\n")
    usable = {p for p, _r, _d in monotone_prefix(cal)}
    print("   %-12s %-12s %-12s %-16s %s"
          % ("period s", "rise s", "rise/period", "peak fin rate", "in the map?"))
    for T, rise, dr in cal:
        print("   %-12.2f %-12.4f %-12.3f %-16s %s"
              % (T, rise, (rise / T if T else float("nan")), "%.4f rad/s" % dr,
                 "yes" if T in usable else "no -- command truncated by its own post-window"))

    print("\n\nA. Guidance-loop speed. tau_f is the seeker-filter lag; T_63 is the measured total")
    print("   loop constant, which is the axis. The airframe alone is %.3f s, so nothing" % loop[0.0])
    print("   faster is reachable on this vehicle.\n")
    out = []
    for tau in TAUS:
        for kind in KINDS:
            sel = [r for r in rows if r["tau_f"] == tau and r["kind"] == kind and r["mid_g"] == 0.0]
            if sel:
                rec = _summarise(sel, tau_f=tau, mid_g=0.0, loop_T63_s=loop[tau], kind=kind)
                rec["period_equiv_s"] = period_equiv(rec["t_rise_s"], cal)
                out.append(rec)
    _print_rows(out, "tau_f", "tau_f")
    print("\n   Controls")
    _print_controls(out, "tau_f", "tau_f")

    print("\n\nB. Midcourse activity, at the fastest loop. Steering during the coast changes the")
    print("   missile's state and therefore the closing geometry and the command the guidance law")
    print("   issues at acquisition, in addition to the content of the pre-command window, so this")
    print("   is a coupled sensitivity test. mid_g = 0 repeats row A exactly, confirming that the")
    print("   code path is shared.\n")
    mid = []
    for g in MIDCOURSE_G:
        sel = [r for r in rows if r["mid_g"] == g and r["tau_f"] == 0.0
               and r["kind"] == "acquisition"]
        if sel:
            rec = _summarise(sel, mid_g=g, tau_f=0.0, loop_T63_s=loop[0.0], kind="acquisition")
            rec["period_equiv_s"] = period_equiv(rec["t_rise_s"], cal)
            mid.append(rec)
    _print_rows(mid, "mid_g", "mid g")
    print("\n   Controls")
    _print_controls(mid, "mid_g", "mid g")
    if len(mid) >= 2:
        a0, a1 = mid[0], mid[-1]
        print("\n   Across this axis the advantage rises")
        print("   (%s -> %s ms) while the muD lead FALLS (%s -> %s ms). The movement is the"
              % (_fmt(a0.get("adv_median"), "%+.0f"), _fmt(a1.get("adv_median"), "%+.0f"),
                 _fmt(a0.get("muD_ms"), "%+.1f"), _fmt(a1.get("muD_ms"), "%+.1f")))
        print("   comparator degrading (%s -> %s ms), because midcourse acceleration contaminates"
              % (_fmt(a0.get("kin_ms"), "%+.1f"), _fmt(a1.get("kin_ms"), "%+.1f")))
        print("   the quiescent window its threshold is calibrated on.")
        print("   Rows with mid_g > 0 are excluded from the onset self-check below.")

    eq = [r["onset_equiv"] for r in rows if r.get("onset_equiv") is not None]
    print("\n\nSelf-check. The trend-relative onset rule should match the project's own rule where")
    print("   the pre-command trend is zero (coasting acquisition events only): %d of %d agree"
          % (sum(1 for x in eq if x), len(eq)))
    print("   within two samples.")
    if eq and not all(eq):
        print("   *** Some events disagree: the trend-relative rule does not reduce to the project's")
        print("   *** rule, and the rows above are invalid.")

    acq = next((r for r in out if r["kind"] == "acquisition" and r["tau_f"] == 0.0), None)
    jnk = [r for r in out if r["kind"] == "jink"]
    print("\n\nSummary. The converting event is seeker acquisition, whose instant is an")
    print("   input to the engagement; its amplitude and shape come from the guidance law. The")
    print("   event whose timing is fully emergent is the jink, at %s detection across the sweep."
          % ("%.0f%%" % (100 * max(r["det"] for r in jnk)) if jnk else "n/a"))
    if acq:
        print("   The result shows the paper's command-duration dependence on an independent trajectory")
        print("   family generated by a guidance law.")

    if args.json:
        # Convert numpy scalars and arrays to plain Python types for json.dump.
        def _plain(o):
            import numpy as _np
            if isinstance(o, (_np.bool_,)):
                return bool(o)
            if isinstance(o, _np.integer):
                return int(o)
            if isinstance(o, _np.floating):
                return float(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            raise TypeError("unserialisable %s" % type(o).__name__)

        # Write to a temporary path and move into place, so a failed dump leaves no partial file at
        # the output path.
        tmp = args.json + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(snr_db=SNR, sigma=SIGMA, reps=REPS, seeds=args.seeds, dwell_s=DWELL,
                           shift_s=SHIFT_S, tau_f_band=list(TAU_F_BAND), t_go_floor=T_GO_FLOOR,
                           taus=list(TAUS), midcourse_g=list(MIDCOURSE_G),
                           attr_det_floor=ATTR_DET_FLOOR,
                           calibration=[dict(period_s=T, t_rise_s=r, d_rate=d) for T, r, d in cal],
                           rows=out, midcourse=mid, per_trajectory=rows),
                      f, indent=2, default=_plain)
        os.replace(tmp, args.json)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
