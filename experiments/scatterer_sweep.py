"""experiments/scatterer_sweep.py -- the last constant on the measurement path nothing varied.

WHY THIS EXISTS. sensitivity_sweep.py swept 13
constants of the MEASUREMENT and none of the PLANT, and the plant turned out to carry the widest
axis in the letter. This is the same defect one level further out, in the SIGNAL MODEL.

Equation (2) of the manuscript writes the return as a SUM over scattering centres,

    s(t) = sum_i a_i exp(j 2 pi phi_i(t)) + n(t),   phi_i = phi_i^0 + (2/lambda) r_i sin delta(t)

with r_i "nonzero only on the control-surface center". That wording asserts the other centres
exist. In the code they do not: dphi_sweep.return_from_fin builds

    s = exp(1j * 4*pi*fin_arm*sin(delta)/lam) + noise

-- ONE unit-amplitude centre whose entire phase follows the fin. With the bulk term removed, every
non-fin centre is a static complex constant added to the fin term, and it dilutes the modulation.
So the printed model is strictly more general than the implemented one, and the difference runs in
the direction that flatters the result. Nothing in the tree varied it.

WHAT IS SWEPT. rho = the fraction of the return's POWER carried by the control-surface centre:

    s(t) = sqrt(rho) exp(j phi_fin(t)) + sqrt(1-rho) + n(t)

rho = 1 is what ships and is a NO-OP: it must reproduce the operating-point cell exactly, and this
script refuses to write if it does not. Everything else is byte-identical to the shipped
measurement -- same seeds, same amp_factor, same 40 dB, same 12 reps, same sigma, same measure().

WHAT IT DECIDES. Two different claims, and they do not survive together:

  * the MEDIAN advantage, which is what the letter reports, and
  * "positive on 30 of 30 trajectories, minimum +11.0 ms", which the letter calls its stable result.

MECHANISM, measured noiseless so no draw enters it: the fin's two-way phase excursion is
4*pi*r*sin(delta)/lambda, which at r = 0.30 m and lambda = 3 cm is several FULL CYCLES. A single
centre with a many-cycle phase sweep is a clean Doppler burst. Add ANY second centre and the
resultant vector encircles the origin once per cycle -- an amplitude null and a phase discontinuity
inside the very window being timed. That is glint, and hughes1998glint is in this letter's own
bibliography. The single-centre model removes it by construction.

    python experiments/scatterer_sweep.py --json runs/ml/scatterer_sweep.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The shipped operating point. Read nothing from a remembered number: these are the same constants
# sensitivity_sweep.py runs at, and the rho = 1 cell must reproduce its baseline.
AMP, SNR, REPS, DWELL, SEEDS, SIGMA = 0.2798, 40.0, 12, 0.002, 30, 0.3

# rho = 1.0 FIRST, because it is the no-op self-check.
RHOS = (1.0, 0.9, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02)


def run_cell(rho):
    """One full n=30 cell at one scatterer power fraction. Patched inside the worker: Windows
    spawns a fresh interpreter per task, and ml imported return_from_fin BY NAME, so the binding
    that has to be replaced is ml's."""
    import numpy as np
    from scipy.stats import wilcoxon
    import experiments.multiclass_lead as ml
    from experiments.multiclass_lead import class_windows, measure
    from experiments.causal_dwell_test import DT_R, C, FC_HZ

    lam = C / FC_HZ

    def return_from_fin(t_fin, delta, snr_db, seed, fin_arm, shift_s=0.0, cue_on=True):
        """dphi_sweep.return_from_fin with the sum given a second, UNMODULATED term.

        Identical noise: p_n is formed from the same stated SNR against unit total signal power, so
        rho moves the modulation depth and nothing else. At rho = 1 this reduces to the shipped
        function line for line, which is what makes the self-check meaningful.
        """
        rng = np.random.default_rng(seed)
        t = np.arange(t_fin[0], t_fin[-1], DT_R)
        if cue_on:
            d = np.interp(t - shift_s, t_fin, delta, left=delta[0], right=delta[-1])
        else:
            d = np.zeros_like(t)
        phase = 4.0 * np.pi * (fin_arm * np.sin(d)) / lam
        s = np.sqrt(rho) * np.exp(1j * phase) + np.sqrt(1.0 - rho)
        p_n = 1.0 / (10.0 ** (snr_db / 10.0))
        s = s + np.sqrt(p_n / 2.0) * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))
        return t, s

    ml.return_from_fin = return_from_fin

    names = [n for n, _ in ml.KIN_ARMS]
    mus, lead, advn, buds, dets, fas = [], {n: [] for n in names}, {n: [] for n in names}, [], [], []
    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                    # noqa: BLE001
            continue
        if not wins:
            continue
        em, el, ea, eb = [], {n: [] for n in names}, {n: [] for n in names}, []
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
            m = measure(w2, SNR, REPS, DWELL, kin_noise=SIGMA)
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
    return dict(rho=rho, n=int(a.size), budget_ms=med(buds), muD_lead_ms=med(mus),
                cusum_lead_ms=med(lead.get("CUSUM Page54", [])), adv_median=med(a),
                worst=float(a.min()) if a.size else None,
                n_pos=int((a > 0).sum()), n_neg=int((a < 0).sum()),
                det=float(np.mean(dets)) if dets else None,
                fa=float(np.mean(fas)) if fas else None,
                adv=[float(x) for x in a],
                p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None)


def phase_excursion():
    """The mechanism, NOISELESS: the fin's two-way phase excursion in full cycles, over the same 30
    trajectories. One encirclement of the origin per cycle for any two-centre target, so this is
    the count of glint nulls the single-centre model removes by construction."""
    import numpy as np
    from experiments.multiclass_lead import (class_windows, drive_airframe, FIN_ARM_M)
    from experiments.causal_dwell_test import DT_R, C, FC_HZ

    lam = C / FC_HZ
    cyc = []
    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                    # noqa: BLE001
            continue
        if not wins:
            continue
        w = dict(wins[0])
        w["a_cmd"] = np.asarray(wins[0]["a_cmd"], float) * AMP
        fl = drive_airframe(w["t"], w["a_cmd"], w["V"], w["alt"])
        t = np.arange(fl["t"][0], fl["t"][-1], DT_R)
        d = np.interp(t, fl["t"], fl["delta"])
        ph = 4.0 * np.pi * (FIN_ARM_M * np.sin(d)) / lam
        cyc.append(float((ph.max() - ph.min()) / (2 * np.pi)))
    return dict(n=len(cyc), cycles_min=min(cyc), cycles_max=max(cyc),
                cycles_median=float(sorted(cyc)[len(cyc) // 2]), lever_arm_m=FIN_ARM_M,
                lambda_m=lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--procs", type=int, default=max(1, min(9, cpu_count() - 2)))
    args = ap.parse_args()

    print("SCATTERER-FRACTION SWEEP -- rho = the fin centre's share of the return power")
    print("  Eq. (2) writes a SUM over centres; the shipped model renders ONE. rho = 1 is that")
    print("  model and is the no-op self-check.\n")
    with Pool(args.procs, maxtasksperchild=1) as pool:
        rows = pool.map(run_cell, list(RHOS), chunksize=1)

    print("%6s %5s %9s %9s %10s %9s %8s %7s"
          % ("rho", "n", "budget", "muD lead", "CUSUM", "ADV", "worst", "det"))
    print("-" * 70)
    for r in rows:
        f = lambda x: (float("nan") if x is None else x)
        print("%6.2f %5d %9.2f %9.2f %10.2f %9.2f %8.1f %6.0f%%"
              % (r["rho"], r["n"], f(r["budget_ms"]), f(r["muD_lead_ms"]), f(r["cusum_lead_ms"]),
                 f(r["adv_median"]), f(r["worst"]), 100 * f(r["det"])))

    base = next(r for r in rows if r["rho"] == 1.0)
    ex = phase_excursion()
    print("\nfin two-way phase excursion, noiseless, over the same %d trajectories: "
          "%.2f-%.2f full cycles (median %.2f)"
          % (ex["n"], ex["cycles_min"], ex["cycles_max"], ex["cycles_median"]))
    print("  every full cycle is one origin encirclement for a two-centre target, so the shipped")
    print("  model removes %.0f-%.0f amplitude nulls from inside the timed window."
          % (ex["cycles_min"], ex["cycles_max"]))

    # SELF-CHECK. rho = 1 IS the shipped signal model, so this cell must reproduce the
    # operating-point cell of the sensitivity sweep exactly. If it does not, the patched
    # return_from_fin is not a strict generalisation of the shipped one and every row above is
    # measuring something else. Fail loudly rather than write a plausible-looking file.
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sp = os.path.join(ROOT, "runs", "ml", "sensitivity_sweep.json")
    if os.path.exists(sp):
        want = json.load(open(sp, encoding="utf-8"))["baseline"]
        for field in ("adv_median", "muD_lead_ms", "cusum_lead_ms", "n"):
            if abs((base[field] or 0) - (want[field] or 0)) > 1e-9:
                raise SystemExit("SELF-CHECK FAILED: rho=1 gives %s=%s, the shipped baseline is %s "
                                 "-- the patch is not a strict generalisation, artifact NOT written"
                                 % (field, base[field], want[field]))
        print("\nself-check: rho=1 reproduces the shipped baseline exactly (adv %+.2f, n=%d)"
              % (base["adv_median"], base["n"]))

    out = dict(rhos=list(RHOS), seeds=SEEDS, reps=REPS, amp_factor=AMP, snr_db=SNR,
               dwell_s=DWELL, sigma=SIGMA, excursion=ex, baseline=base, cells=rows)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1, default=float)
        print("wrote %s" % args.json)


if __name__ == "__main__":
    main()
