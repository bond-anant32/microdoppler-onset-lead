"""experiments/endgame_lead.py -- the lead, measured on commands a guidance law produced.

WHAT THIS ANSWERS. The letter's duration axis is swept by imposing a command shape on a class whose
generator samples at 0.5 s, because a command shorter than a sample cannot come out of that
generator. trajectory_generators/endgame.py removes that constraint: it flies a proportional-
navigation engagement at 1e-4 s, so the command's amplitude and shape are outputs of the guidance
law and the closing geometry. This runs the letter's own
detector, comparator, threshold rule and null on that command. Read the next section before
concluding anything from it -- what remains chosen is the one thing that decides how much the
result is worth.

WHAT THIS DOES AND DOES NOT ESTABLISH. Read this before quoting any row.

    It is NOT evidence that a maneuvering target commands inside the letter's duration bound. Of the
    two events an engagement produces, the one that converts is seeker ACQUISITION, and the instant
    at which the seeker acquires is an input to the engagement rather than an output of it. Its
    amplitude and shape are the guidance law's; its timing is not. Quoting it as target-maneuver
    onset would repeat, one level up, the substitution this file exists to remove.

    The event whose timing IS fully emergent is the missile's answer to the target's break, and it
    does not convert. That is the finding. An independent trajectory family, sourced from a guidance
    law rather than from an imposed waveform, reproduces the letter's bound instead of escaping it.

THE AXIS IS A MEASURED LOOP CONSTANT, NOT A CHOSEN DURATION. Zarchan (Science and Global Security
8(1):99-124, 1999) works his examples at guidance-system time constants of 0.05, 0.1, 0.2 and 0.5 s
and attributes the dominant part of the total, endoatmospherically, to the flight control system; he
states no fielded range and none is claimed here. This project resolves that flight control system,
so the sweep varies only the seeker-filter remainder and reports the MEASURED total from
endgame.loop_time_constant(). The airframe alone already measures 0.186 s, so the sweep spans
0.186-0.479 s and says nothing about faster loops.

TWO EVENTS PER ENGAGEMENT, AND THEY TEST DIFFERENT THINGS.

    acquisition  the seeker closes the loop on a vehicle that has been coasting unguided. The
                 pre-command window is quiescent because the engagement made it so. This is the
                 endgame's analogue of the commanded step the letter reports, with the same
                 weakness: the onset instant is chosen.
    jink         the missile answers the target's break. Nothing about this event's timing is
                 chosen -- the target breaks, the line-of-sight rate answers, and the command
                 follows. It is the one fully emergent maneuver onset available here.

CONTROLS, ALL THREE REPORTED PER CELL.

    null         the fin held still, identical noise, identical threshold rule. A detection rate
                 that is not separated from its own null is an alarm rate.
    attribution  the fin history delayed by 40 ms and nothing else changed. If the alarm does not
                 move by 40 ms it is not being timed by the fin. This is the control that disposed
                 of the phase-displacement arm, and it is reported here alongside the fraction of
                 the fin's travel that already falls inside the window setting the threshold. It is
                 only meaningful where the cell detects: below ATTR_DET_FLOOR the alarm is formed
                 from a handful of marginal realizations and its movement is noise, so the row is
                 marked rather than read.
    chain        the same decision taken by thresholding the fin history directly -- no radar
                 rendering, no receiver noise, no statistic. This does NOT discriminate a
                 guidance-sourced cue from a laundered command: at 40 dB the chain is invertible
                 with respect to the decision either way, which is the same thing
                 experiments/chain_removal_test.py reports for the shipped cue. What it does show is
                 that no result here is an artifact of the signal processing.

THE COMPARATOR ARM IS UNTOUCHED. measure() is called with two of its inputs replaced -- the fin
history, and the onset rule -- and nothing else: the comparator is still Page CUSUM on TRUE lateral
acceleration plus noise, which never estimates the state it thresholds and is therefore an oracle,
so every advantage below is a lower bound against a real tracker exactly as in the letter. Both arms
are timed against the same substituted onset, so the substitution cannot favour one of them. The
run's own SELF-CHECK asserts that the substituted onset rule reduces to the project's where the
pre-command trend is zero, which is the case the letter reports.

    python experiments/endgame_lead.py --json runs/ml/endgame_lead.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Matched to the letter's shipped operating point (runs/ml/shape_period.json, true_shape.json,
# phase_cue.json all carry snr 40, sigma 0.3, reps 12, seeds 30) so the rows are comparable.
SNR, REPS, SIGMA, SEEDS, DWELL = 40.0, 12, 0.3, 30, 0.002
SHIFT_S = 0.040                       # the attribution delay, as in phase_cue_detector.py
TAUS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
KINDS = ("acquisition", "jink")
# Midcourse lateral activity, in g, run at the fastest loop only -- the only cell where the cue
# converts, so the only one where removing it means anything. 0.0 repeats the sweep's own row and
# is the control: if it does not reproduce that row exactly, the axis is not isolated.
MIDCOURSE_G = (0.0, 0.25, 0.5, 1.0, 2.0)
ATTR_DET_FLOOR = 0.25   # below this detection rate the attribution median is formed from too few
                        # alarms to read; see _summarise.


def _trend_onset(ev):
    """The project's departure rule, applied to the departure rather than to the absolute level.

    multiclass_lead.onset_from_achieved thresholds |a_z| at 2.0 m/s^2, which is the right question
    for a vehicle that was not already accelerating. A missile answering a target break is: it is
    part-way through taking out its heading error, so its |a_z| is already above the threshold when
    the window opens and the rule would return the first sample. The same 2.0 m/s^2 is therefore
    applied to the residual against the trend the achieved acceleration was already on, which is
    the identical question and reduces to the identical answer at acquisition, where that trend is
    zero. The trend is fitted to the ACHIEVED acceleration over the pre-command window and never to
    the command.
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

        # TWO of measure()'s inputs are replaced, not one. The flown fin history replaces the
        # re-driven one, AND the onset rule is replaced by its trend-relative form -- which moves
        # t_on, and therefore the budget and the threshold window along with it. The rendering, the
        # statistic, the threshold RULE, the comparator and the null are untouched.
        #
        # The onset substitution is a generalisation, so it has to reduce to the original where the
        # original applies. On the acquisition event the pre-command trend is identically zero, so
        # the two rules must return the same t_on; `equiv` records whether they did, and the
        # self-check below asserts it.
        ml.drive_airframe = lambda *_a, **_k: dict(t=tf, delta=delta, az=az, cmd=ev["a_cmd"])
        ml.onset_from_achieved = _trend_onset(ev)
        # Equivalence is only claimed where the pre-command trend really is zero: the acquisition
        # event of an engagement that coasted. Under midcourse steering the vehicle IS already
        # accelerating when the window opens, so the native rule fires on that and the two rules
        # SHOULD disagree -- that disagreement is the reason the trend rule exists, and scoring it
        # as a failure would report the fix as the defect.
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

        # The letter's advantage is paired against Page CUSUM (KIN_ARMS[1]), not against the
        # trailing mean that measure() puts in its top-level `adv`. Quoting the top-level figure
        # here would compare this experiment with a different comparator from the one the letter
        # reports. Every arm is carried so the choice is visible and none has to be re-run for.
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
    """Measure the published duration axis with the same clock this experiment uses.

    experiments/shape_period_sweep.py reports detection against the PERIOD of a commanded sinusoid.
    endgame.command_timescale reports a rise measured from a 2 m/s^2 departure. Converting one to
    the other analytically as period/4 is wrong, because the sinusoid's quarter-period is measured
    from zero and the rise is not -- the offset depends on amplitude, and it flatters whichever
    side it is applied to. So the sinusoids the sweep actually flies are run through the identical
    function, at the identical amplitude, and the result is a measured map.

    Returns [(period_s, t_rise_s, d_rate)] over shape_period_sweep.PERIODS.
    """
    import numpy as np
    import experiments.multiclass_lead as ml
    from experiments.shape_period_sweep import PERIODS, AMP
    from trajectory_generators.endgame import command_timescale, _departure

    out = []
    for T in PERIODS:
        # The amplitude is the one the sweep flies: the sampled command scaled by AMP. Taken from
        # the same class_windows call the sweep makes, so no amplitude is invented here.
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
    """The part of the calibration that is actually invertible.

    multiclass_lead.synth_command sizes its post-window from where the command reaches three times
    the departure threshold and caps it at 3 s, so a long-period sinusoid is cut off before it
    peaks and its MEASURED rise stops tracking its period: 0.241 of the period at 0.1-1.0 s, then
    0.191, 0.091, 0.044 as the cap bites, and not even monotone. A map built over the whole range
    would read a truncation as a duration. Only the strictly increasing prefix is kept, and rises
    outside it get no equivalent period rather than a clamped one.
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
    # A cell that barely detects produces its alarm median from a few marginal realizations, and
    # the movement of that median under the fin delay is then noise about a number nobody should
    # read. Marked, not deleted: a suppressed row is indistinguishable from a row that was never
    # run, and the reader is owed the difference.
    rec["attr_readable"] = bool(rec["det"] >= ATTR_DET_FLOOR and rec.get("moved_ms") is not None)
    # The same floor decides whether the ADVANTAGE can be read, for the reason the letter already
    # gives about its own sweep: a pair is formed only where both arms alarm, so under the floor
    # the surviving pairs are the trajectories the outcome selected, and their median is a
    # statement about that selection. Marked, not hidden -- the cells still ran.
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

    print("ENDGAME LEAD -- the letter's detector on commands proportional navigation produced")
    print("  guidance at 1e-4 s; the command is an output of the law, not a chosen waveform")
    print("  SNR %.0f dB | dwell %.0f ms | sigma %.2f | %d noise reps | %d trajectories per cell"
          % (SNR, 1000 * DWELL, SIGMA, REPS, args.seeds))
    print("  advantage is paired against Page CUSUM, as in the letter")
    print("  engagement not modeled inside t_go = %.2f s\n" % T_GO_FLOOR)

    jobs = ([(t, s, 0.0) for t in TAUS for s in range(args.seeds)]
            + [(0.0, s, g) for g in MIDCOURSE_G[1:] for s in range(args.seeds)])
    with Pool(args.procs) as pool:
        got = pool.map(run_cell, jobs)
    rows = [r for chunk in got for r in chunk]
    loop = {t: loop_time_constant(t)[0] for t in TAUS}

    cal = calibrate_periods()
    print("0. AXIS CALIBRATION. The published duration sweep's own sinusoids, measured with this")
    print("   experiment's clock, so the two axes are related by measurement and not by a factor.\n")
    usable = {p for p, _r, _d in monotone_prefix(cal)}
    print("   %-12s %-12s %-12s %-16s %s"
          % ("period s", "rise s", "rise/period", "peak fin rate", "in the map?"))
    for T, rise, dr in cal:
        print("   %-12.2f %-12.4f %-12.3f %-16s %s"
              % (T, rise, (rise / T if T else float("nan")), "%.4f rad/s" % dr,
                 "yes" if T in usable else "no -- command truncated by its own post-window"))

    print("\n\nA. GUIDANCE-LOOP SPEED. tau_f is the seeker-filter lag; T_63 is the MEASURED total")
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
    print("\n   CONTROLS")
    _print_controls(out, "tau_f", "tau_f")

    print("\n\nB. MIDCOURSE ACTIVITY, at the fastest loop. This is a COUPLED sensitivity test, not")
    print("   an isolated one: steering during the coast changes the missile's state, hence the")
    print("   closing geometry, hence the command the guidance law issues at acquisition, as well")
    print("   as what the pre-command window contains. mid_g = 0 repeats row A exactly, which")
    print("   shows only that the code path is shared.\n")
    mid = []
    for g in MIDCOURSE_G:
        sel = [r for r in rows if r["mid_g"] == g and r["tau_f"] == 0.0
               and r["kind"] == "acquisition"]
        if sel:
            rec = _summarise(sel, mid_g=g, tau_f=0.0, loop_T63_s=loop[0.0], kind="acquisition")
            rec["period_equiv_s"] = period_equiv(rec["t_rise_s"], cal)
            mid.append(rec)
    _print_rows(mid, "mid_g", "mid g")
    print("\n   CONTROLS")
    _print_controls(mid, "mid_g", "mid g")
    if len(mid) >= 2:
        a0, a1 = mid[0], mid[-1]
        print("\n   READ THE COLUMNS, NOT THE ADVANTAGE. Across this axis the advantage rises")
        print("   (%s -> %s ms) while the muD lead FALLS (%s -> %s ms). The movement is the"
              % (_fmt(a0.get("adv_median"), "%+.0f"), _fmt(a1.get("adv_median"), "%+.0f"),
                 _fmt(a0.get("muD_ms"), "%+.1f"), _fmt(a1.get("muD_ms"), "%+.1f")))
        print("   comparator degrading (%s -> %s ms), because midcourse acceleration contaminates"
              % (_fmt(a0.get("kin_ms"), "%+.1f"), _fmt(a1.get("kin_ms"), "%+.1f")))
        print("   the quiescent window ITS threshold is calibrated on as well. A larger number here")
        print("   is a worse experiment, not a better cue.")

    eq = [r["onset_equiv"] for r in rows if r.get("onset_equiv") is not None]
    print("\n\nSELF-CHECK. The trend-relative onset rule must reduce to the project's own rule where")
    print("   the pre-command trend is zero (coasting acquisition events only): %d of %d agree"
          % (sum(1 for x in eq if x), len(eq)))
    print("   within two samples.")
    if eq and not all(eq):
        print("   *** THEY DO NOT. The onset substitution is not a generalisation; do not read the")
        print("   *** rows above until this is resolved.")

    acq = next((r for r in out if r["kind"] == "acquisition" and r["tau_f"] == 0.0), None)
    jnk = [r for r in out if r["kind"] == "jink"]
    print("\n\nWHAT THIS SUPPORTS. The converting event is seeker ACQUISITION, whose instant is an")
    print("   input to the engagement; only its amplitude and shape are the guidance law's. The")
    print("   event whose timing is fully emergent is the jink, at %s detection across the sweep."
          % ("%.0f%%" % (100 * max(r["det"] for r in jnk)) if jnk else "n/a"))
    if acq:
        print("   So this is evidence FOR the letter's duration bound on an independent trajectory")
        print("   family, not evidence that a maneuvering target commands inside it.")

    if args.json:
        # A numpy scalar that reaches json.dump aborts the write AFTER every table has printed, so
        # the run looks complete and leaves no artifact. Coerced rather than trusted.
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

        # Written to a temporary path and moved into place. json.dump streams, so a failure part
        # way through leaves a TRUNCATED file at the real path -- which parses as nothing but
        # exists, so an artifact check finds it and a reader sees a run that never completed.
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
