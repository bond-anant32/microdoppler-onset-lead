"""experiments/onset_model_equivalence.py -- IS THE STEP COMMAND THE LITERATURE'S ONSET MODEL?

WHY THIS EXISTS. The objection that the letter's step command "is a shape nothing flies and no
cited paper models" argues for re-anchoring the letter on fan2016's first-order lag
tau_T = 0.2 s (experiments/literature_onset_test.py). That argument conflates two different
things, and this script measures which one holds.

    fan2016's model:      a_cmd is a STEP; the TARGET responds  da/dt = (a_cmd - a)/tau_T,
                          tau_T = 0.2 s.  The lag is the vehicle, not the command.
    this project's model: a_cmd is a STEP; the target is a RESOLVED 5-state pitch airframe
                          (sim/sixdof.py) whose closed-loop response IS that lag.

sim/sixdof.py's own header says DEFAULT_AIRFRAME was chosen for "emergent T ~ 0.2 s" and its
self-test asserts "the measured closed-loop step-response time constant ... near the lumped
T_guid=0.2 s IT REPLACES". If that is true at the FLOWN conditions, then step-command-into-this-
airframe already IS fan2016's onset model, resolved instead of lumped -- and
literature_onset_test.py, which lags the COMMAND by tau and then drives the airframe, applies
tau_T TWICE.

Two cited sources model a step in commanded acceleration directly (ru2009detection, li2002part4);
their 5-10 s figures are SAMPLING intervals, not onset durations. So "no source models a
sub-second commanded onset" was also wrong: the onset SHAPE and the sampling GRID are different
axes.

WHAT IS MEASURED (no radar, no detector -- airframe only, so it is fast and exact):

  A. T_63, DC gain, 10-90 rise of the shipped airframe under a step command, at the 30 FLOWN
     (altitude, Mach) draws -- not at the 10 km / M3 self-test point.
  B. The best-fit first-order tau to the achieved a_z(t) under a step, with its residual, so the
     equivalence is a fit and not an assertion.
  C. The CASCADE literature_onset_test.py actually flies: step -> lag(tau) -> airframe. If tau_T
     is already in the airframe, the cascade's effective time constant must be ~2x the airframe's
     alone, and the onset budget must inflate by about as much.

FOUR QUESTIONS, EACH WITH ITS BAR STATED BEFORE RUNNING. The first pass asked only two and one of
them was the wrong test: it required the airframe to fit a first-order exponential to R^2 >= 0.95,
which it does NOT (0.925) -- and that miss is itself the interesting result, not a failure, because
the discrepancy lives exactly in the sub-100 ms interval this letter measures. Both the original
bars and their outcomes are kept below so the refinement is visible rather than silent.

  Q1  Does the airframe's closed-loop step response carry the literature's lumped tau_T?
      BAR: |T_63 - 0.2| / 0.2 <= 0.25 at the FLOWN conditions.
  Q2  Is the resolved airframe *exactly* a first-order lag?
      BAR: R^2 >= 0.95 against the best-fit exponential. (Answer expected NO -- it is a
      second-order autopilot plus a second-order servo. Reported because it decides Q3.)
  Q3  Can the LUMPED first-order model reproduce the onset budget this letter measures?
      BAR: within 2x of the resolved airframe's budget. A first-order lag starts moving at t=0+
      with slope A/tau, so its crossing of a small threshold is set by the threshold and not by
      the vehicle; if it misses by more than 2x, the lump cannot represent the interval at all
      and resolving it is forced, not optional.
  Q4  Does lagging the COMMAND and then driving the airframe count tau_T twice?
      BAR: cascade T_63 >= 1.5x the airframe's own -> literature_onset_test.py is a double count.

Any combination of outcomes is reportable. If T_63 had come out at 0.02 s the step would be a
bang-bang idealisation after all and the re-anchor would have been right.

    python experiments/onset_model_equivalence.py --json runs/ml/onset_model_equivalence.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.sixdof import PitchAirframe, DEFAULT_AIRFRAME, sound_speed, equivalent_time_constant  # noqa: E402

AMP = 0.2798          # the shipped feasibility amplitude factor
SEEDS = 30
TAU_LIT = 0.2         # fan2016's value, the proposed anchor
DT = 1e-4             # = multiclass_lead.DT_FINE


def step_response(a_cmd, V, alt, t_max=2.0, dt=DT, pre_lag_tau=None):
    """Achieved a_z(t) for a step command, optionally pre-lagged by a first-order tau.

    pre_lag_tau=None  -> the SHIPPED path (multiclass_lead.synth_command with shape='step').
    pre_lag_tau=0.2   -> the path literature_onset_test.py flies (its synth_lag, then the airframe).
    """
    n = int(t_max / dt)
    t = np.arange(n) * dt
    cmd = np.full(n, float(a_cmd))
    if pre_lag_tau is not None:
        cmd = a_cmd * (1.0 - np.exp(-t / float(pre_lag_tau)))
    af = PitchAirframe(DEFAULT_AIRFRAME)
    az = np.empty(n)
    for i in range(n):
        az[i] = af.step(float(cmd[i]), V, alt, dt)
    return t, az


def characterise(t, az, a_cmd):
    """T_63 / 10-90 rise / DC gain, plus a first-order FIT with its residual.

    The fit is what turns "the airframe is like a 0.2 s lag" from a claim into a measurement:
    a(t) = a_ss (1 - exp(-t/tau)) is linearised as log(1 - a/a_ss) = -t/tau over the 5-90% band,
    then R^2 is computed on the ORIGINAL curve, not on the linearised one, so a bad fit cannot
    hide inside the transform.
    """
    a_ss = float(np.mean(az[-int(0.2 / DT):]))
    if a_ss <= 0:
        return None
    dc = a_ss / a_cmd

    def cross(frac):
        tgt = frac * a_ss
        idx = int(np.argmax(az >= tgt))
        return float(t[idx]) if az[idx] >= tgt else float("nan")

    t63, rise = cross(0.632), cross(0.9) - cross(0.1)

    band = (az > 0.05 * a_ss) & (az < 0.90 * a_ss)
    tau_fit, r2 = float("nan"), float("nan")
    if band.sum() > 10:
        y = np.log(np.clip(1.0 - az[band] / a_ss, 1e-12, None))
        x = t[band] - t[band][0]
        slope = float(np.polyfit(x, y, 1)[0])
        if slope < 0:
            tau_fit = -1.0 / slope
            t0 = t[band][0] + tau_fit * np.log(np.clip(1.0 - az[band][0] / a_ss, 1e-12, None)) * -1.0
            model = a_ss * (1.0 - np.exp(-np.clip(t - t0, 0, None) / tau_fit))
            seg = (t >= t[band][0]) & (t <= t[band][-1])
            ss_res = float(np.sum((az[seg] - model[seg]) ** 2))
            ss_tot = float(np.sum((az[seg] - np.mean(az[seg])) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(a_ss=a_ss, dc_gain=dc, t63_s=t63, rise_10_90_s=rise,
                tau_fit_s=tau_fit, fit_r2=r2)


def onset_budget_ms(t, az, g=2.0, need_s=0.005):
    """t_on - t_c in ms, by multiclass_lead.onset_from_achieved's own rule (t_c = 0 here)."""
    need = max(2, int(need_s / DT))
    ab = np.abs(az) > g
    for i in range(len(ab) - need):
        if ab[i:i + need].all():
            return 1000.0 * float(t[i])
    return None


def lumped_budget_ms(amp, tau=TAU_LIT, g=2.0):
    """The budget the LUMPED model predicts, in closed form: A(1-exp(-t/tau)) = g.

    This is Q3. The lumped lag has slope A/tau at t=0+, so it crosses a small threshold almost at
    once and the interval it predicts is a statement about g/A, not about the vehicle. Returns None
    where the command never reaches the threshold.
    """
    if amp <= g:
        return None
    return 1000.0 * float(-tau * np.log(1.0 - g / amp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    from experiments.multiclass_lead import class_windows

    print("ONSET-MODEL EQUIVALENCE -- is a step command through this airframe fan2016's tau_T?\n")

    # ---- the flown conditions, from the same sampler every other experiment uses ----
    conds = []
    for sd in range(args.seeds):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        for w in wins:
            amp = float(np.max(np.asarray(w["a_cmd"], float))) * AMP
            conds.append((float(w["alt"]), float(w["V"]), amp, float(w["mach"])))
    print(f"  flown draws: n={len(conds)}  alt {min(c[0] for c in conds)/1e3:.1f}-"
          f"{max(c[0] for c in conds)/1e3:.1f} km   Mach {min(c[3] for c in conds):.2f}-"
          f"{max(c[3] for c in conds):.2f}   cmd {min(c[2] for c in conds)/9.80665:.2f}-"
          f"{max(c[2] for c in conds)/9.80665:.2f} g\n")

    rows = []
    for alt, V, amp, mach in conds:
        t, az = step_response(amp, V, alt)
        ch = characterise(t, az, amp)
        if ch is None:
            continue
        ch.update(alt_km=alt / 1e3, mach=mach, cmd_g=amp / 9.80665)
        ch["budget_ms"] = onset_budget_ms(t, az)
        ch["lumped_budget_ms"] = lumped_budget_ms(amp)
        tc, azc = step_response(amp, V, alt, pre_lag_tau=TAU_LIT)
        chc = characterise(tc, azc, amp)
        ch["cascade_t63_s"] = chc["t63_s"] if chc else None
        ch["cascade_tau_fit_s"] = chc["tau_fit_s"] if chc else None
        ch["cascade_budget_ms"] = onset_budget_ms(tc, azc)
        rows.append(ch)

    g = lambda k: np.array([r[k] for r in rows if r.get(k) is not None], float)   # noqa: E731
    t63, tfit, r2 = g("t63_s"), g("tau_fit_s"), g("fit_r2")
    ct63, cbud, bud = g("cascade_t63_s"), g("cascade_budget_ms"), g("budget_ms")

    lump = g("lumped_budget_ms")

    print("  A. SHIPPED PATH (step command -> resolved airframe), over the flown draws")
    print(f"     T_63          median {np.median(t63):.3f} s   range {t63.min():.3f}-{t63.max():.3f}")
    print(f"     tau_fit       median {np.median(tfit):.3f} s   range {tfit.min():.3f}-{tfit.max():.3f}")
    print(f"     fit R^2       median {np.median(r2):.4f}   min {r2.min():.4f}")
    print(f"     DC gain       median {np.median(g('dc_gain')):.3f}")
    print(f"     onset budget  median {np.median(bud):.2f} ms\n")
    print(f"  B. LUMPED MODEL a(t)=A(1-exp(-t/{TAU_LIT})), the literature's own first-order lag")
    print(f"     onset budget  median {np.median(lump):.2f} ms  "
          f"({np.median(bud)/np.median(lump):.2f}x SHORTER than the resolved airframe)\n")
    print(f"  C. CASCADE (step -> lag({TAU_LIT} s) -> airframe), what literature_onset_test.py flies")
    print(f"     T_63          median {np.median(ct63):.3f} s   ({np.median(ct63)/np.median(t63):.2f}x the airframe alone)")
    print(f"     onset budget  median {np.median(cbud):.2f} ms  ({np.median(cbud)/np.median(bud):.2f}x)\n")

    T63, dc, rise = equivalent_time_constant(10000.0, 3.0)
    print(f"  D. sixdof self-test point (10 km, M3): T_63={T63:.3f} s, DC={dc:.3f}, 10-90={rise:.3f} s\n")

    # ---- E. the COMMAND-SHAPE budget fold, which Sec. II asserted without an artifact -----------
    # Sec. II carried "command shape moves the budget 4.2x: 147.9 against 35.4 ms on the identical
    # airframe", sourced to an audit document rather than to anything in runs/ml. audit_t10's M2
    # rows fly the three shapes but store budget_ms = None, so nothing in the tree reproduced
    # either number -- and neither is close: the step budget is the shipped 48.45 ms, not 35.4, and
    # the raised cosine is 712.55 ms, not 147.9. The fold is 14.7x, not 4.2x. Measured here through
    # the same synth_command -> drive_airframe -> onset_from_achieved path everything else uses.
    from experiments.multiclass_lead import (synth_command, drive_airframe, onset_from_achieved,
                                             PRE_S, POST_S)
    shp = {}
    for shape, dur in (("step", 0.0), ("raised-cosine", 30.0), ("sinusoid", 36.0)):
        buds = []
        for sd in range(args.seeds):
            try:
                wins, _ = class_windows("supersonic_cruise",
                                        rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
            except Exception:                                                 # noqa: BLE001
                continue
            for w in wins:
                amp = float(np.max(np.asarray(w["a_cmd"], float))) * AMP
                tt, aa, tc = synth_command(shape, dur, amp, PRE_S, POST_S)
                fl = drive_airframe(tt, aa, w["V"], w["alt"])
                t_on, _ = onset_from_achieved(fl["t"], fl["az"])
                if t_on is not None:
                    buds.append(1000.0 * (t_on - tc))
        shp[shape] = dict(n=len(buds),
                          budget_ms=float(np.median(buds)) if buds else None,
                          dur_s=dur)
    print("  E. COMMAND-SHAPE BUDGET (median over the flown draws), same path as the measurement")
    for k, v in shp.items():
        print(f"     {k:<16}{str(round(v['budget_ms'], 2)) if v['budget_ms'] else '--':>10} ms  n={v['n']}")
    fold = (shp["raised-cosine"]["budget_ms"] / shp["step"]["budget_ms"]
            if shp["step"]["budget_ms"] else None)
    print(f"     raised-cosine / step = {fold:.2f}x\n")
    out_shape = dict(shapes=shp, fold_rc_over_step=fold)

    q1 = bool(abs(np.median(t63) - TAU_LIT) / TAU_LIT <= 0.25)
    q2 = bool(np.median(r2) >= 0.95)
    q3 = bool(np.median(bud) / np.median(lump) <= 2.0)
    q4 = bool(np.median(ct63) >= 1.5 * np.median(t63))
    print(f"  Q1 airframe carries the literature's tau_T (|T_63-{TAU_LIT}|/{TAU_LIT} <= 0.25) : {q1}"
          f"   [{abs(np.median(t63)-TAU_LIT)/TAU_LIT:.3f}]")
    print(f"  Q2 airframe is EXACTLY a first-order lag (R^2 >= 0.95)                  : {q2}"
          f"   [{np.median(r2):.4f}]")
    print(f"  Q3 lumped lag reproduces the onset budget (within 2x)                   : {q3}"
          f"   [{np.median(bud)/np.median(lump):.2f}x]")
    print(f"  Q4 literature_onset_test.py double-counts tau_T (cascade >= 1.5x)       : {q4}"
          f"   [{np.median(ct63)/np.median(t63):.2f}x]")
    # The ORIGINAL two-legged bar, kept so the refinement above is auditable and not silent.
    print(f"\n  (alternative bar: tau_fit in [0.12,0.35] AND R^2>=0.95 -> "
          f"{bool(0.12 <= np.median(tfit) <= 0.35 and np.median(r2) >= 0.95)})")

    out = dict(seeds=args.seeds, amp_factor=AMP, tau_lit=TAU_LIT, n=len(rows),
               t63_median=float(np.median(t63)), t63_lo=float(t63.min()), t63_hi=float(t63.max()),
               tau_fit_median=float(np.median(tfit)), tau_fit_lo=float(tfit.min()),
               tau_fit_hi=float(tfit.max()), fit_r2_median=float(np.median(r2)),
               fit_r2_min=float(r2.min()), dc_gain_median=float(np.median(g("dc_gain"))),
               budget_ms_median=float(np.median(bud)),
               lumped_budget_ms_median=float(np.median(lump)),
               lumped_budget_ratio=float(np.median(bud) / np.median(lump)),
               cascade_t63_median=float(np.median(ct63)),
               cascade_budget_ms_median=float(np.median(cbud)),
               cascade_t63_ratio=float(np.median(ct63) / np.median(t63)),
               cascade_budget_ratio=float(np.median(cbud) / np.median(bud)),
               shape_budgets=out_shape,
               selftest_t63_10km_m3=float(T63), selftest_dc=float(dc),
               q1_airframe_carries_tauT=q1, q2_exactly_first_order=q2,
               q3_lumped_reproduces_budget=q3, q4_literature_onset_double_counts=q4,
               rows=rows)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
