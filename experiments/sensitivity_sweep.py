"""experiments/sensitivity_sweep.py -- sensitivity of the headline advantage to each free constant.

Each cell reruns the n=30 supersonic_cruise measurement with one constant perturbed and records the
budget, the muD, CUSUM and GLR leads, the paired advantage over CUSUM, and detection and
false-alarm rates. The knobs cover the measurement path (k_frac, glr_window, dwell, onset_g,
fin_arm, need_ms, grid, search_pre, cruise_window, guard_band, snr, amp_factor) and the airframe
plant (sim/sixdof.DEFAULT_AIRFRAME).

Hand-chosen constants in the model that can carry the result:

    KIN_NOISE = 0.3            sets the headline; the advantage changes sign at 0
    GLR window = dwell/DT_R    makes the Willsky-Jones GLR a monotone transform of the trailing-mean arm
    q-bar reference condition  the feasibility scalar is derived at 52.4 kPa and applied at ~95
    np.arange(0.0, 25.1, 0.5)  an alpha bound 5 deg past the model's stall limit, which selects
                               1/3.57 over 1/4.7

Cells that set a knob to its default value must reproduce the baseline exactly; otherwise the
script exits without writing the artifact.

    python experiments/sensitivity_sweep.py --json runs/ml/sensitivity_sweep.json
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AMP, SNR, REPS, DWELL, SEEDS, SIGMA = 0.2798, 40.0, 12, 0.002, 30, 0.3

# (knob, value). Each knob's sweep includes its default value, so that cell can be compared with the
# baseline in the same run.
CELLS = (
    [("baseline", 0.0)]
    + [("k_frac", v) for v in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.3)]
    + [("glr_window", v) for v in (4, 10, 20, 40, 100, 200, 400)]
    + [("dwell", v) for v in (0.0005, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050)]
    + [("onset_g", v) for v in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)]
    + [("fin_arm", v) for v in (0.05, 0.15, 0.30, 0.60, 1.00)]
    + [("need_ms", v) for v in (2, 3, 5, 10)]
    + [("grid", v) for v in (0.0005, 0.001, 0.002, 0.005)]
    + [("search_pre", v) for v in (0.15, 0.25, 0.40)]
    + [("cruise_window", v) for v in (0.30, 0.60)]
    + [("guard_band", v) for v in (0.06, 0.12)]
    + [("snr", v) for v in (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)]
    + [("amp_factor", v) for v in (0.2128, 0.2798, 0.4931)]
    # SNR crossed with the over-driven amplitude (1.0). The noise variance of unfiltered phase
    # differentiation scales as sigma_n^2/dt_R^2 and does not depend on how hard the target
    # manoeuvres. If detection holds well below 40 dB at amplitude 1.0, the 40 dB requirement comes
    # from the amplitude; if it collapses at low SNR, the floor comes from the differentiation noise.
    + [("snr_overdrive", v) for v in (10.0, 15.0, 20.0, 25.0, 40.0)]
    # ---------------------------------------------------------------- airframe plant
    # The knobs above are constants of the measurement. The rows below perturb the airframe that
    # generates both the cue and the budget, sim/sixdof.DEFAULT_AIRFRAME (a canonical SM-class
    # interceptor). At 30/30 pairing and 97-100% detection the advantage reaches +5.0 ms at
    # CN_delta = 1.2 and +44.5 ms at a 50 deg/s rate limit.
    #
    # CN_delta sets direct fin lift: C_N_delta*delta moves a_z at once, which is why a noiseless
    # comparator alarms at t_c (Section 2 of the paper). 0.3 is near-pure moment control; 1.2 is
    # typical of canard or wing control. wn_servo spans 5-150 rad/s, extended to 300.
    + [("CN_delta", v) for v in (0.0, 0.05, 0.15, 0.30, 0.60, 1.20)]
    + [("wn_servo", v) for v in (5.0, 10.0, 25.0, 50.0, 100.0, 150.0, 300.0)]
    + [("zeta_servo", v) for v in (0.30, 0.65, 1.20)]
    + [("rate_max_dps", v) for v in (50.0, 100.0, 200.0, 400.0, 800.0)]
    + [("CN_alpha", v) for v in (5.75, 8.0, 11.5, 16.0, 23.0)]
    + [("wn_autopilot", v) for v in (3.0, 10.0, 20.0)]
)

# knob -> (DEFAULT_AIRFRAME field, swept value -> field value, field value -> swept units).
# The no-op self-check in main() uses the inverse transform to read each default from the airframe.
PLANT = {
    "CN_delta":     ("CN_delta", lambda v: v,          lambda x: x),
    "wn_servo":     ("wn_s",     lambda v: v,          lambda x: x),
    "zeta_servo":   ("zeta_s",   lambda v: v,          lambda x: x),
    "rate_max_dps": ("rate_max", math.radians,         math.degrees),
    "CN_alpha":     ("CN_alpha", lambda v: v,          lambda x: x),
    "wn_autopilot": ("wn_cl",    lambda v: v,          lambda x: x),
}


def run_cell(cell):
    """One full n=30 cell with exactly one constant perturbed. Patched inside the worker because
    Windows spawns a fresh interpreter per task."""
    knob, val = cell
    import numpy as np
    from scipy.stats import wilcoxon
    import experiments.class_profiles as cp
    import experiments.causal_dwell_test as cdt
    from experiments.causal_dwell_test import causal_lead as _cl

    if knob == "onset_g":                      # must precede multiclass_lead's import-time binding
        cp.ONSET_G_MS2 = val
    import experiments.multiclass_lead as ml
    from experiments.multiclass_lead import class_windows, measure, stat_cusum, stat_glr

    amp, snr, dwell = AMP, SNR, DWELL
    if knob == "onset_g":
        ml.ONSET_G_MS2 = val
    elif knob == "k_frac":
        ml.KIN_ARMS = (("trailing-mean", None),
                       ("CUSUM Page54", lambda x, d: stat_cusum(x, d, k_frac=val)),
                       ("GLR Willsky76", stat_glr))
    elif knob == "glr_window":
        ml.KIN_ARMS = (("trailing-mean", None), ("CUSUM Page54", stat_cusum),
                       ("GLR Willsky76", lambda x, d: stat_glr(x, d, win=int(val))))
    elif knob == "fin_arm":
        ml.FIN_ARM_M = val
    elif knob == "need_ms":
        ml.causal_lead = lambda t, s, thr, t_on: _cl(t, s, thr, t_on, need_ms=val)
    elif knob == "search_pre":
        ml.causal_lead = lambda t, s, thr, t_on: _cl(t, s, thr, t_on, search_pre=val)
    elif knob == "grid":
        cdt.DECISION_GRID_S = val
    elif knob in ("cruise_window", "guard_band"):
        lo = val if knob == "cruise_window" else 0.60
        hi = val if knob == "guard_band" else 0.12

        def thr(t, stat, t_on, fpr=None):
            m = np.isfinite(stat) & (t >= t_on - lo) & (t <= t_on - hi)
            return float(np.max(stat[m])) if m.sum() > 8 else np.inf
        ml.thr_from_cruise = thr
    elif knob == "snr":
        snr = val
    elif knob == "snr_overdrive":
        snr, amp = val, 1.0
    elif knob == "dwell":
        dwell = val
    elif knob == "amp_factor":
        amp = val
    elif knob in PLANT:
        # multiclass_lead imports DEFAULT_AIRFRAME by name, so the patch goes on ml.DEFAULT_AIRFRAME,
        # the binding drive_airframe reads. The dict is copied so a shared object cannot leak
        # across workers.
        field, to_af, _ = PLANT[knob]
        af = dict(ml.DEFAULT_AIRFRAME)
        af[field] = to_af(val)
        ml.DEFAULT_AIRFRAME = af

    names = [n for n, _ in ml.KIN_ARMS]
    mus, lead, advn, buds, dets, fas = [], {n: [] for n in names}, {n: [] for n in names}, [], [], []
    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=amp)
        except Exception:                                                  # noqa: BLE001
            continue
        if not wins:
            continue
        em, el, ea, eb = [], {n: [] for n in names}, {n: [] for n in names}, []
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * amp
            m = measure(w2, snr, REPS, dwell, kin_noise=SIGMA)
            if not m:
                continue
            eb.append(m["budget_ms"]); dets.append(m["det"]); fas.append(m["fa"])
            if m.get("muD") is not None:
                em.append(m["muD"])
            for nm in names:
                r = (m.get("arms") or {}).get(nm) or {}
                if r.get("lead") is not None:
                    el[nm].append(r["lead"])
                if r.get("adv") is not None:
                    ea[nm].append(r["adv"])
        if em:
            mus.append(float(np.median(em)))
        if eb:
            buds.append(float(np.median(eb)))
        for nm in names:
            if el[nm]:
                lead[nm].append(float(np.median(el[nm])))
            if ea[nm]:
                advn[nm].append(float(np.median(ea[nm])))

    a = np.asarray(advn.get("CUSUM Page54", []), float)
    med = lambda v: float(np.median(v)) if len(v) else None
    return dict(knob=knob, value=val, n=int(a.size),
                budget_ms=med(buds), muD_lead_ms=med(mus),
                cusum_lead_ms=med(lead.get("CUSUM Page54", [])),
                glr_lead_ms=med(lead.get("GLR Willsky76", [])),
                adv_median=med(a), n_pos=int((a > 0).sum()), n_neg=int((a < 0).sum()),
                worst=float(a.min()) if a.size else None,
                det=float(np.mean(dets)) if dets else None,
                fa=float(np.mean(fas)) if fas else None,
                p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--procs", type=int, default=max(1, min(12, cpu_count() - 2)))
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated knobs to run (default: all)")
    args = ap.parse_args()
    cells = CELLS if not args.only else \
        [c for c in CELLS if c[0] in set(args.only.split(",")) or c[0] == "baseline"]

    print("SENSITIVITY SWEEP -- %d cells over %d constants, n=%d each, sigma=%.1f\n"
          % (len(cells), len({k for k, _ in cells}) - 1, SEEDS, SIGMA))
    # maxtasksperchild=1 is required for correctness. Each cell patches module globals in place
    # (ml.KIN_ARMS, ml.causal_lead, ml.thr_from_cruise, cp.ONSET_G_MS2), and Pool otherwise reuses
    # a worker process across tasks, carrying one cell's patches into the next. One task per
    # process keeps the cells isolated.
    with Pool(args.procs, maxtasksperchild=1) as pool:
        rows = pool.map(run_cell, cells, chunksize=1)

    base = next(r for r in rows if r["knob"] == "baseline")
    cells = [r for r in rows if r["knob"] != "baseline"]

    # Self-check: several cells set a knob to its default value. These no-op cells must reproduce
    # the baseline exactly; a mismatch indicates cross-worker contamination (see maxtasksperchild
    # above), and the script exits without writing the artifact.
    NOOP = {"k_frac": 0.5, "dwell": 0.002, "onset_g": 2.0, "fin_arm": 0.30, "need_ms": 3,
            "grid": 0.001, "search_pre": 0.25, "cruise_window": 0.60, "guard_band": 0.12,
            "snr": 40.0, "amp_factor": 0.2798}
    # Plant defaults are read from DEFAULT_AIRFRAME, so the no-op values follow sim/sixdof.py.
    from sim.sixdof import DEFAULT_AIRFRAME                                   # noqa: E402
    for k, (field, _to_af, from_af) in PLANT.items():
        NOOP[k] = from_af(DEFAULT_AIRFRAME[field])
    # Float-tolerant match: plant defaults round-trip through radians, and degrees(radians(400.0))
    # differs from 400.0 in the last bits, so exact equality would miss that no-op cell.
    def is_noop(r):
        want = NOOP.get(r["knob"])
        return want is not None and abs(float(r["value"]) - float(want)) <= 1e-9 * max(1.0, abs(want))

    n_noop = sum(1 for r in cells if is_noop(r))
    expected_noop = len(NOOP)
    bad = [(r["knob"], r["adv_median"]) for r in cells
           if is_noop(r) and r["adv_median"] is not None
           and abs(r["adv_median"] - base["adv_median"]) > 1e-9]
    if bad:
        print("\nSelf-check failed: no-op cells do not reproduce the baseline %+.2f ms"
              % base["adv_median"])
        for k, v in bad:
            print("      %-14s returned %+.2f" % (k, v))
        raise SystemExit("worker processes interfered; artifact not written")
    if not args.only and n_noop != expected_noop:
        raise SystemExit("self-check incomplete: %d no-op cells matched, %d knobs declare one. A "
                         "default value lies outside its sweep, so that knob has no no-op cell."
                         % (n_noop, expected_noop))
    print("self-check: %d no-op cells all reproduce the baseline %+.2f ms"
          % (n_noop, base["adv_median"]))
    knobs = sorted({r["knob"] for r in cells})
    advs = [r["adv_median"] for r in cells if r["adv_median"] is not None]
    invariant = [k for k in knobs
                 if len({round(r["adv_median"], 6) for r in cells
                         if r["knob"] == k and r["adv_median"] is not None}) == 1]

    print("%-14s %8s %6s %9s %9s %9s %8s" %
          ("knob", "value", "n", "muD", "CUSUM", "advantage", "worst"))
    print("-" * 70)
    for k in knobs:
        for r in [x for x in cells if x["knob"] == k]:
            print("%-14s %8g %6d %9.2f %9.2f %9.2f %8.1f"
                  % (r["knob"], r["value"], r["n"], r["muD_lead_ms"] or float("nan"),
                     r["cusum_lead_ms"] or float("nan"), r["adv_median"] or float("nan"),
                     r["worst"] if r["worst"] is not None else float("nan")))
    print("\n%d constants, %d cells: advantage spans %+.2f to %+.2f ms, sign %s"
          % (len(knobs), len(cells), min(advs), max(advs),
             "positive throughout" if min(advs) > 0 else "changes sign"))
    print("exactly invariant: %s" % ", ".join(invariant))

    out = dict(sigma=SIGMA, seeds=SEEDS, reps=REPS, amp_factor=AMP, snr_db=SNR, dwell_s=DWELL,
               n_constants=len(knobs), n_cells=len(cells), knobs=knobs,
               adv_min=min(advs), adv_max=max(advs), all_positive=bool(min(advs) > 0),
               exactly_invariant=invariant, baseline=base, cells=cells)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
