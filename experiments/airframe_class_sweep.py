"""experiments/airframe_class_sweep.py -- the axis the letter says it cannot sweep.

WHAT THIS ATTACKS. DEFAULT_AIRFRAME supplies the budget *and* the a_max_at gate that excludes
MaRV. Its constants are swept, but the CHOICE of airframe is a different axis, and the
manuscript's conclusion states that the magnitude is set by the airframe more than by the channel.
This measures that axis instead of leaving it as an unquantified limitation.

It is sweepable. What makes it hard is not the sweep, it is the CONFOUND: a different vehicle
differs in divert authority AND in response speed at once, and divert authority already has an
axis (the amplitude sweep, and a_max_at gates which events are admitted at all). Changing both
together measures their sum and attributes it to neither.

SO THE FAMILY IS DESIGNED TO HOLD AUTHORITY FIXED AND VARY RESPONSE SPEED. Every member must pass
the same three consistency anchors sim/sixdof.py states for the shipped airframe --

    a_max(10 km, M3) = qbar*Sref*CN_max/m        divert authority
    AoA_max          = CN_max/CN_alpha           incidence limit
    DC gain          = 1                         no steady acceleration bias

-- to within a stated tolerance, while the emergent closed-loop response T_63 spans a decade. The
vehicles are otherwise physically distinct: diameter, mass, pitch inertia, all four aero
derivatives, servo bandwidth and fin rate limit all move, coupled the way a real scaling would
couple them (m ~ d^3 at constant density, Iyy ~ m L^2, short-period w_sp = sqrt(-M_alpha)).

WHY THIS IS NOT THE EXISTING SWEEPS. sensitivity_sweep moves wn_cl, wn_s, rate_max, CN_delta,
CN_alpha and zeta_s ONE AT A TIME around the shipped point; joint_mc_sweep draws four of them at
once but still around that point. Neither ever visits a coherent DIFFERENT VEHICLE -- a 1.2 m,
6 t airframe has an inertia the shipped 0.34 m, 200 kg one cannot reach by perturbing a knob, and
its autopilot cannot be fast because its short period is not.

THE FAMILY BRACKETS THE CITED RANGE, WHICH IS THE POINT. The maneuvering-target papers this letter
cites that state a target response constant state three different values -- fan2016 tau_T = 0.2 s
(Table 2, p. 7), oshman2006 tau_T = 0.2 s (Table I, p. 320), oshman2004 tau_T = 0.4 s (Table 1,
p. 600) -- all under the same first-order form da/dt = (a^c - a)/tau driven by a bang-bang command.
Measured (--anchors), this family lands:

    agile 0.089 s | interceptor 0.186 s | hcv 0.371 s | heavy 0.895 s

so "interceptor" sits on fan2016/oshman2006's value and "hcv" sits on oshman2004's, and the two
ends are outside the cited range in both directions. The shipped airframe is at the FAST end of
what the literature assumes, and that is a conservatism question this sweep answers rather than
concedes.

THE CONTROL IS LOAD-BEARING. Member "interceptor" IS sim/sixdof.py's DEFAULT_AIRFRAME, and it must
reproduce the shipped headline exactly. If it does not, the workers contaminated each other
through the shared DEFAULT_AIRFRAME dict this script mutates in place, and every other row is
void. Same rule sensitivity_sweep's 17 no-op cells enforce.

    python experiments/airframe_class_sweep.py --anchors            # design check only, seconds
    python experiments/airframe_class_sweep.py --json runs/ml/airframe_class_sweep.json
"""
import argparse
import copy
import json
import math
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AMP, SNR, REPS, DWELL, SIGMA, SEEDS = 0.2798, 40.0, 12, 0.002, 0.3, 30


def _af(d, L, m, Iyy, CN_alpha, CN_max, CN_delta, Cm_alpha, Cm_delta, Cm_q,
        wn_s, zeta_s, delta_max_deg, rate_max_dps, wn_cl, zeta_cl):
    return dict(d=d, Sref=math.pi * d ** 2 / 4.0, L=L, m=m, Iyy=Iyy, Ixx=Iyy / 100.0,
                CN_alpha=CN_alpha, CN_max=CN_max, CN_delta=CN_delta,
                Cm_alpha=Cm_alpha, Cm_delta=Cm_delta, Cm_q=Cm_q,
                wn_s=wn_s, zeta_s=zeta_s,
                delta_max=math.radians(delta_max_deg), rate_max=math.radians(rate_max_dps),
                wn_cl=wn_cl, zeta_cl=zeta_cl)


# Four vehicles, ordered by emergent response time. Authority anchors held; everything structural
# moves. Values are LABELLED SURROGATES in the same sense sim/sixdof.py's are -- representative of
# a class, not a specification of any vehicle.
#
#  agile        a small, high-bandwidth airframe: the fastest end anything airbreathing reaches
#  interceptor  THE SHIPPED AIRFRAME -- the control, must reproduce the headline
#  hcv          a hypersonic cruise vehicle: the class these trajectories actually come from,
#               ~4x the diameter and ~6x the mass, so its short period is slower and its autopilot
#               cannot be faster than its short period
#  heavy        a large aircraft-scale maneuvering target, the slow bound
AIRFRAMES = [
    ("agile",       _af(0.20, 2.4,    90.0,     42.0, 13.0, 5.00, 0.45,
                        -22.0, -30.0, -120.0, 250.0, 0.65, 25.0, 900.0, 20.0, 0.70)),
    ("interceptor", _af(0.34, 4.3,   200.0,    262.0, 11.5, 4.00, 0.30,
                        -13.8, -16.0, -200.0, 150.0, 0.65, 20.0, 400.0, 10.0, 0.70)),
    ("hcv",         _af(0.70, 8.0,   900.0,   6500.0, 12.2, 4.25, 0.24,
                        -5.2,  -6.0, -900.0,  70.0, 0.68, 22.0, 150.0,  5.0, 0.72)),
    # Two geometric interpolants between hcv and heavy. They exist for ONE reason: the first pass
    # found the advantage rising monotonically to +26.0 ms at hcv and then the cue failing outright
    # at heavy, which brackets the boundary only between 0.37 and 0.90 s -- too loose to say
    # anything useful about where a real vehicle stops converting. Every structural quantity is the
    # geometric mean at 1/3 and 2/3 of the way, so the authority anchors carry over exactly rather
    # than being re-tuned (Sref*CN_max/m is 1.816e-3 for all six members, to four figures).
    ("interp-a",    _af(0.815, 9.16, 1284.0, 13341.0, 12.87, 4.47, 0.2097,
                        -3.84, -4.42, -1479.0, 54.0, 0.687, 23.0, 115.8, 3.92, 0.73)),
    ("interp-b",    _af(0.948, 10.48, 1831.0, 27381.0, 13.57, 4.71, 0.1831,
                        -2.84, -3.26, -2432.0, 41.7, 0.693, 24.0,  89.4, 3.07, 0.74)),
    ("heavy",       _af(1.10, 12.0, 2600.0,  56000.0, 14.3, 4.97, 0.16,
                        -2.1,  -2.4, -4000.0, 32.0, 0.70, 25.0,  70.0,  2.4, 0.75)),
]

# Anchor tolerances, stated before running. a_max and AoA_max are what must be HELD; T_63 is what
# must MOVE, and a family that does not span at least 4x in it has not tested anything.
A_MAX_G_BAND = (20.0, 45.0)       # divert authority at 10 km / M3, in g
AOA_MAX_BAND = (18.0, 26.0)       # incidence limit, deg
DC_GAIN_TOL = 0.05
T63_SPAN_MIN = 4.0


def _anchors(af):
    from sim.sixdof import a_max_at, aoa_max_deg, equivalent_time_constant, G0
    t63, dc, rise = equivalent_time_constant(10000.0, 3.0, airframe=af)
    return dict(a_max_g=a_max_at(10000.0, 3.0, airframe=af) / G0,
                aoa_max_deg=aoa_max_deg(af), t63_s=t63, dc_gain=dc, rise_10_90_s=rise)


def _install(af):
    """Mutate the SHARED DEFAULT_AIRFRAME dict in place -- the single patch point.

    sim.sixdof.a_max_at / aoa_max_deg / equivalent_time_constant bind DEFAULT_AIRFRAME as a DEFAULT
    ARGUMENT at def time, and experiments/class_profiles.py imports a_max_at by name, so rebinding
    the module global would miss them. Mutating the dict object itself reaches every consumer:
    PitchAirframe.__init__ copies with dict(airframe), so it must be done before construction.

    Per-process, in a Pool initializer, with one pool per airframe -- so no worker ever sees two.
    """
    import sim.sixdof as sd
    sd.DEFAULT_AIRFRAME.clear()
    sd.DEFAULT_AIRFRAME.update(af)


def _init(af):
    _install(af)


def run_seed(sd_i):
    import numpy as np
    from experiments.multiclass_lead import class_windows, measure

    try:
        wins, _ = class_windows("supersonic_cruise",
                                rng=np.random.default_rng(90000 + sd_i), amp_factor=AMP)
    except Exception:                                                         # noqa: BLE001
        return None
    if not wins:
        return None
    ea, em, ek, dets, fas, buds = [], [], [], [], [], []
    for w in wins:
        w2 = dict(w)
        w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
        m = measure(w2, SNR, REPS, DWELL, kin_noise=SIGMA)
        if not m:
            continue
        dets.append(m["det"]); fas.append(m["fa"]); buds.append(m["budget_ms"])
        if m.get("muD") is not None:
            em.append(m["muD"])
        r = (m.get("arms") or {}).get("CUSUM Page54") or {}
        if r.get("lead") is not None:
            ek.append(r["lead"])
        if r.get("adv") is not None:
            ea.append(r["adv"])
    med = lambda v: float(np.median(v)) if len(v) else None                   # noqa: E731
    # det/fa are POOLED over realizations, not median-over-seeds. The letter reports 97% as
    # 349 alarms of 360 realizations, and a median of per-seed fractions is a different quantity
    # that reads as 100% wherever most seeds are 12/12 -- printing it beside the letter's 97% for
    # the SAME configuration would be a visible inconsistency for no reason.
    return dict(adv=med(ea), muD=med(em), kin=med(ek), budget=med(buds),
                det_hits=float(np.sum(dets)), det_n=len(dets),
                fa_hits=float(np.sum(fas)), fa_n=len(fas), n_windows=len(dets))


def race(name, af, procs):
    import numpy as np
    from scipy.stats import wilcoxon

    with Pool(processes=procs, initializer=_init, initargs=(af,), maxtasksperchild=1) as pool:
        got = pool.map(run_seed, range(SEEDS))
    got = [g for g in got if g]
    a = np.asarray([g["adv"] for g in got if g["adv"] is not None], float)
    col = lambda k: [g[k] for g in got if g.get(k) is not None]               # noqa: E731
    med = lambda v: float(np.median(v)) if len(v) else None                   # noqa: E731
    dh, dn = sum(g["det_hits"] for g in got), sum(g["det_n"] for g in got)
    fh, fn = sum(g["fa_hits"] for g in got), sum(g["fa_n"] for g in got)
    return dict(airframe=name, n_seeds=len(got), n=int(a.size),
                n_windows=sum(g["n_windows"] for g in got),
                adv_median=med(a), worst=float(a.min()) if a.size else None,
                n_pos=int((a > 0).sum()), muD_lead_ms=med(col("muD")),
                cusum_lead_ms=med(col("kin")), budget_ms=med(col("budget")),
                det=(dh / dn) if dn else None, fa=(fh / fn) if fn else None,
                p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--anchors", action="store_true", help="design check only, no race")
    ap.add_argument("--procs", type=int, default=max(1, min(12, cpu_count() - 2)))
    args = ap.parse_args()

    print("AIRFRAME-CLASS SWEEP -- authority held, response speed varied over a decade\n")
    print(f"  {'airframe':<12}{'a_max@10km/M3':>15}{'AoA_max':>10}{'T_63':>9}{'DC':>7}{'10-90':>8}")
    anch, bad = {}, []
    for name, af in AIRFRAMES:
        _install(af)
        a = _anchors(af)
        anch[name] = a
        print(f"  {name:<12}{a['a_max_g']:>13.1f} g{a['aoa_max_deg']:>9.1f}d"
              f"{a['t63_s']:>8.3f}s{a['dc_gain']:>7.3f}{a['rise_10_90_s']:>7.3f}s")
        if not (A_MAX_G_BAND[0] <= a["a_max_g"] <= A_MAX_G_BAND[1]):
            bad.append(f"{name}: a_max {a['a_max_g']:.1f} g outside {A_MAX_G_BAND}")
        if not (AOA_MAX_BAND[0] <= a["aoa_max_deg"] <= AOA_MAX_BAND[1]):
            bad.append(f"{name}: AoA_max {a['aoa_max_deg']:.1f} deg outside {AOA_MAX_BAND}")
        if abs(a["dc_gain"] - 1.0) > DC_GAIN_TOL:
            bad.append(f"{name}: DC gain {a['dc_gain']:.3f} not unity")
    t63s = [anch[n]["t63_s"] for n, _ in AIRFRAMES]
    span = max(t63s) / min(t63s)
    print(f"\n  T_63 span {span:.2f}x  ({min(t63s):.3f}-{max(t63s):.3f} s)")
    if span < T63_SPAN_MIN:
        bad.append(f"T_63 span {span:.2f}x below the {T63_SPAN_MIN}x this sweep exists to test")
    if bad:
        print("\n  DESIGN FAILS ITS OWN ANCHORS -- not racing:")
        for b in bad:
            print("    " + b)
        # Refuse to write, the way both shipped sweeps refuse: a family that does not hold the
        # anchors measures authority and calls it response speed.
        sys.exit(2)
    print("  anchors held.\n")
    if args.anchors:
        return

    _install(copy.deepcopy(dict(AIRFRAMES[1][1])))       # restore the shipped one in THIS process
    rows = []
    for name, af in AIRFRAMES:
        r = race(name, af, args.procs)
        r.update(anch[name])
        rows.append(r)
        fmt = lambda v: ("%+.2f" % v) if v is not None else "  --  "         # noqa: E731
        print(f"  {name:<12} T63 {r['t63_s']:.3f}s  adv {fmt(r['adv_median']):>7}  "
              f"worst {fmt(r['worst']):>8}  n_pos {r['n_pos']:>2}/{r['n']:<2} "
              f"(win {r['n_windows']:>2})  det {100*(r['det'] or 0):>3.0f}%  "
              f"fa {100*(r['fa'] or 0):.0f}%  budget {fmt(r['budget_ms']):>7}  "
              f"muD {fmt(r['muD_lead_ms']):>7}")

    ctrl = next(r for r in rows if r["airframe"] == "interceptor")
    full = [r for r in rows if r["n"] == SEEDS and r["adv_median"] is not None]
    allpos = all(r["adv_median"] > 0 for r in full)
    print(f"\n  CONTROL (interceptor, = DEFAULT_AIRFRAME): adv {ctrl['adv_median']}, "
          f"det {100*(ctrl['det'] or 0):.0f}%, budget {ctrl['budget_ms']} "
          f"-- must equal the shipped headline, else workers contaminated each other")
    print(f"  sign positive in all full-pairing airframes: {allpos} ({len(full)} of {len(rows)})")

    out = dict(seeds=SEEDS, reps=REPS, snr_db=SNR, sigma=SIGMA, amp_factor=AMP,
               a_max_band_g=list(A_MAX_G_BAND), aoa_band_deg=list(AOA_MAX_BAND),
               t63_span=float(span), control=ctrl["adv_median"],
               all_full_pairing_positive=bool(allpos), rows=rows)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
