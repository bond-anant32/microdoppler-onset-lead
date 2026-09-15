"""experiments/scatterer_sweep.py -- lead advantage versus the fin centre's share of return power.

Eq. (2) of the paper writes the return as a sum over scattering centres,

    s(t) = sum_i a_i exp(j 2 pi phi_i(t)) + n(t),   phi_i = phi_i^0 + (2/lambda) r_i sin delta(t)

with r_i nonzero only on the control-surface centre. dphi_sweep.return_from_fin renders a single
unit-amplitude centre whose phase follows the fin,

    s = exp(1j * 4*pi*fin_arm*sin(delta)/lam) + noise

With the bulk term removed, each non-fin centre adds a static complex constant to the fin term and
dilutes the modulation. This script sweeps rho, the fraction of return power carried by the
control-surface centre,

    s(t) = sqrt(rho) exp(j phi_fin(t)) + sqrt(1-rho) + n(t)

rho = 1 is the single-centre model. The script checks that this cell reproduces the baseline in
runs/ml/sensitivity_sweep.json and exits without writing if it does not. Seeds, amp_factor, SNR
(40 dB), repetitions (12), sigma and measure() are those of the reported operating point. Each cell
reports the median paired advantage over CUSUM, the worst trajectory, and the numbers of positive
and negative trajectories.

The fin's two-way phase is 4*pi*r*sin(delta)/lambda, with r = 0.30 m and lambda = 3 cm, and
phase_excursion() measures its excursion in full cycles, without noise, over the 30 trajectories.
A single centre gives a clean Doppler burst. With a second centre the resultant is a rotating fin phasor plus a static one, and
each cycle of fin phase produces an amplitude fade and a phase jump inside the timed window, both
deepest when the two powers are equal. This is glint [hughes1998glint].

    python experiments/scatterer_sweep.py --json runs/ml/scatterer_sweep.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Operating point, the same constants as the sensitivity_sweep.py baseline.
AMP, SNR, REPS, DWELL, SEEDS, SIGMA = 0.2798, 40.0, 12, 0.002, 30, 0.3

# Fin-centre power fractions; rho = 1.0 is the single-centre model.
RHOS = (1.0, 0.9, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02)


def run_cell(rho):
    """Run the 30-trajectory cell at fin-centre power fraction rho and return its summary dict.

    The patch is applied inside the worker because Windows spawns a fresh interpreter per task.
    multiclass_lead imports return_from_fin by name, so ml.return_from_fin is the binding replaced.
    """
    import numpy as np
    from scipy.stats import wilcoxon
    import experiments.multiclass_lead as ml
    from experiments.multiclass_lead import class_windows, measure
    from experiments.causal_dwell_test import DT_R, C, FC_HZ

    lam = C / FC_HZ

    def return_from_fin(t_fin, delta, snr_db, seed, fin_arm, shift_s=0.0, cue_on=True):
        """dphi_sweep.return_from_fin with a second, unmodulated centre of power 1 - rho.

        Noise power p_n is set from snr_db against unit total signal power, so rho changes only
        the modulation depth. At rho = 1 the function equals dphi_sweep.return_from_fin.
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
    """Noiseless two-way phase excursion of the fin, in full cycles, over the same 30 trajectories.

    For a two-centre target each full cycle produces one amplitude fade. Returns n, cycles_min,
    cycles_max, cycles_median, lever_arm_m and lambda_m.
    """
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
    print("  Eq. (2) writes a sum over centres; the default signal model renders one. rho = 1 is")
    print("  that model and reproduces the sensitivity-sweep baseline.\n")
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
    print("  each full cycle is one amplitude fade for a two-centre target, so the single-centre")
    print("  model removes %.0f-%.0f amplitude nulls from inside the timed window."
          % (ex["cycles_min"], ex["cycles_max"]))

    # rho = 1 is the single-centre signal model, so its cell must equal the operating-point cell
    # of the sensitivity sweep. On a mismatch the patched return_from_fin does not generalise
    # dphi_sweep.return_from_fin, and the script exits before writing the artifact.
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sp = os.path.join(ROOT, "runs", "ml", "sensitivity_sweep.json")
    if os.path.exists(sp):
        want = json.load(open(sp, encoding="utf-8"))["baseline"]
        for field in ("adv_median", "muD_lead_ms", "cusum_lead_ms", "n"):
            if abs((base[field] or 0) - (want[field] or 0)) > 1e-9:
                raise SystemExit("self-check failed: rho=1 gives %s=%s, the default baseline is %s; "
                                 "artifact not written"
                                 % (field, base[field], want[field]))
        print("\nself-check: rho=1 reproduces the default baseline exactly (adv %+.2f, n=%d)"
              % (base["adv_median"], base["n"]))

    out = dict(rhos=list(RHOS), seeds=SEEDS, reps=REPS, amp_factor=AMP, snr_db=SNR,
               dwell_s=DWELL, sigma=SIGMA, excursion=ex, baseline=base, cells=rows)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1, default=float)
        print("wrote %s" % args.json)


if __name__ == "__main__":
    main()
