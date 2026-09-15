"""experiments/onset_model_equivalence.py -- resolved-airframe step response vs the lumped lag.

In fan2016 the commanded acceleration a_cmd is a step and the target responds as a first-order lag,
da/dt = (a_cmd - a)/tau_T with tau_T = 0.2 s. In this code a_cmd is also a step, and the target is
a resolved 5-state pitch airframe (sim/sixdof.py) whose closed-loop response supplies the lag.
sim/sixdof.py selects DEFAULT_AIRFRAME for an emergent time constant near the lumped 0.2 s. When
that holds at the flown conditions, a step into this airframe is fan2016's onset model with the lag
resolved, and lagging the command by tau before driving the airframe (the cascade flown by
literature_onset_test.py) applies tau_T twice.

ru2009detection and li2002part4 also model a step in commanded acceleration; their 5-10 s figures
are sampling intervals, a separate axis from onset shape.

Measured on the airframe alone (no radar or detector):

  A. T_63, DC gain and 10-90 rise under a step command at the 30 flown (altitude, Mach) draws.
  B. Best-fit first-order tau to the achieved a_z(t), with its R^2.
  C. The cascade step -> lag(tau) -> airframe: T_63 and onset budget relative to the airframe alone.
  D. The sixdof self-test point (10 km, M3).
  E. Onset budget for step, raised-cosine and sinusoid commands.

Questions and pass criteria:

  Q1  Does the closed-loop step response carry the lumped tau_T?
      |T_63 - 0.2| / 0.2 <= 0.25 at the flown conditions.
  Q2  Is the resolved airframe exactly a first-order lag?
      R^2 >= 0.95 against the best-fit exponential. The airframe is a second-order autopilot plus
      a second-order servo; its median R^2 is 0.925, and the departure lies in the sub-100 ms
      interval the paper measures.
  Q3  Does the lumped first-order model reproduce the onset budget?
      Within 2x of the resolved airframe's budget. A first-order lag moves at t=0+ with slope
      A/tau, so its threshold crossing time -tau*ln(1 - g/A) contains none of the servo,
      short-period or rate-limit dynamics.
  Q4  Does the cascade count tau_T twice?
      Cascade T_63 >= 1.5x the airframe's own.

    python experiments/onset_model_equivalence.py --json runs/ml/onset_model_equivalence.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.sixdof import PitchAirframe, DEFAULT_AIRFRAME, sound_speed, equivalent_time_constant  # noqa: E402

AMP = 0.2798          # feasible command-amplitude factor
SEEDS = 30
TAU_LIT = 0.2         # fan2016's first-order target time constant (s)
DT = 1e-4             # = multiclass_lead.DT_FINE


def step_response(a_cmd, V, alt, t_max=2.0, dt=DT, pre_lag_tau=None):
    """Achieved a_z(t) for a step command, optionally pre-lagged by a first-order tau.

    pre_lag_tau=None  -> the path multiclass_lead uses (synth_command with shape='step').
    pre_lag_tau=0.2   -> the cascade: command lagged by tau, then the airframe.
    Returns (t, a_z).
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
    """T_63, 10-90 rise and DC gain, plus a first-order fit with its R^2.

    a(t) = a_ss (1 - exp(-t/tau)) is linearised as log(1 - a/a_ss) = -t/tau over the 5-90% band.
    R^2 is computed on the original curve, so the log transform does not mask a poor fit.
    Returns None if the steady-state a_ss is not positive.
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
    """t_on - t_c in ms by the multiclass_lead.onset_from_achieved rule (t_c = 0 here), or None."""
    need = max(2, int(need_s / DT))
    ab = np.abs(az) > g
    for i in range(len(ab) - need):
        if ab[i:i + need].all():
            return 1000.0 * float(t[i])
    return None


def lumped_budget_ms(amp, tau=TAU_LIT, g=2.0):
    """Onset budget (ms) of the lumped model in closed form, solving A(1 - exp(-t/tau)) = g.

    Used for Q3. The result, -tau*ln(1 - g/A), depends only on tau and g/A. Returns None when
    amp <= g, where the command never reaches the threshold.
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

    print("Onset-model equivalence: step command through the resolved airframe vs fan2016's tau_T\n")

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

    print("  A. STEP PATH (step command -> resolved airframe), over the flown draws")
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

    # ---- E. onset budget by command shape --------------------------------------------------------
    # Measured through the synth_command -> drive_airframe -> onset_from_achieved path that
    # multiclass_lead uses.
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
    # Alternative combined criterion on the first-order fit: tau_fit in [0.12, 0.35] s and R^2 >= 0.95.
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
