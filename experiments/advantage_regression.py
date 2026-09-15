"""Regression of the muD-over-CUSUM advantage on the sampled (commanded g, altitude, Mach) triple.

Produces the advantage r^2 reported in Section 2 of the paper. For each of the 30 flown trajectories the
commanded g, altitude and Mach come from the shared class sampler, and the paired advantage against
CUSUM is measured at the paper's operating point (sigma = 0.3 m/s^2, 40 dB, amplitude factor 0.2798).
The per-trajectory medians are regressed by ordinary least squares with an intercept on the nested
models

    g            ->  r^2
    g + alt      ->  r^2
    g + alt + M  ->  r^2

The command-to-onset budget is regressed on the same draws, which reproduces M10 of
experiments/audit_t10.py; when runs/ml/audit_t10.json is present the script reports whether the two
budget r^2 values agree within 0.02. The budget r^2 and the advantage r^2 are different quantities.

    python experiments/advantage_regression.py --json runs/ml/advantage_regression.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AMP, SNR, REPS, DWELL, SIGMA, SEEDS = 0.2798, 40.0, 12, 0.002, 0.3, 30


def run_seed(sd):
    import numpy as np
    from experiments.multiclass_lead import class_windows, measure

    try:
        wins, _ = class_windows("supersonic_cruise",
                                rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
    except Exception:                                                         # noqa: BLE001
        return None
    if not wins:
        return None
    advs, buds, gs, alts, machs = [], [], [], [], []
    for w in wins:
        w2 = dict(w)
        w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
        m = measure(w2, SNR, REPS, DWELL, kin_noise=SIGMA)
        if not m:
            continue
        r = (m.get("arms") or {}).get("CUSUM Page54") or {}
        if r.get("adv") is None:
            continue
        advs.append(float(r["adv"]))
        buds.append(float(m["budget_ms"]))
        gs.append(float(w["amp_g"]) * AMP)
        alts.append(float(w["alt"]))
        machs.append(float(w["mach"]))
    if not advs:
        return None
    med = lambda v: float(np.median(v))                                       # noqa: E731
    return dict(adv=med(advs), budget=med(buds), g=med(gs), alt=med(alts), mach=med(machs))


def r2(y, X):
    """OLS r^2 of y on the columns of X with an intercept, as in M10 of audit_t10.py. NaN if y is constant."""
    import numpy as np
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--procs", type=int, default=max(1, min(12, cpu_count() - 2)))
    args = ap.parse_args()

    print("Advantage regression on (commanded g, altitude, Mach)\n")
    with Pool(processes=args.procs, maxtasksperchild=1) as pool:
        got = [g for g in pool.map(run_seed, range(SEEDS)) if g]
    adv = [g["adv"] for g in got]
    bud = [g["budget"] for g in got]
    G = [g["g"] for g in got]
    AL = [g["alt"] for g in got]
    MA = [g["mach"] for g in got]
    print("  n = %d trajectories" % len(got))
    print("  advantage  median %+.2f ms   range %+.1f to %+.1f"
          % (np.median(adv), min(adv), max(adv)))

    out = dict(seeds=SEEDS, reps=REPS, snr_db=SNR, sigma=SIGMA, amp_factor=AMP, n=len(got))
    print("\n  advantage regressed on the sampled triple")
    for name, X in (("g", [G]), ("g+alt", [G, AL]), ("g+alt+mach", [G, AL, MA])):
        v = r2(adv, X)
        out["adv_r2_" + name.replace("+", "_")] = float(v)
        print("    adv ~ %-14s r2 = %.4f" % (name, v))
    print("\n  budget regressed on the same triple (cross-check with audit_t10 M10)")
    for name, X in (("g", [G]), ("g+alt", [G, AL]), ("g+alt+mach", [G, AL, MA])):
        v = r2(bud, X)
        out["budget_r2_" + name.replace("+", "_")] = float(v)
        print("    budget ~ %-11s r2 = %.4f" % (name, v))

    t10 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runs", "ml", "audit_t10.json")
    if os.path.exists(t10):
        m10 = (json.load(open(t10, encoding="utf-8")).get("M10") or {})
        ref = m10.get("amp_g+alt+mach")
        if ref is not None:
            gap = abs(out["budget_r2_g_alt_mach"] - ref)
            print("\n  CONTROL: audit_t10 M10 budget r2 = %.4f, here %.4f, gap %.4f -> %s"
                  % (ref, out["budget_r2_g_alt_mach"], gap,
                     "reproduces" if gap < 0.02 else "does not reproduce"))
            out["control_m10_ref"] = float(ref)
            out["control_reproduces"] = bool(gap < 0.02)

    out.update(rows=got)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print("\n  wrote %s" % args.json)


if __name__ == "__main__":
    main()
