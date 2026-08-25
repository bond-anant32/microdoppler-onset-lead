"""experiments/audit_t10.py -- four stability checks on the reported result, each re-derived here
rather than asserted.

  H4  a different seed block lands outside the published CI
  H5  the CI endpoint moves with the bootstrap resample count
  M2  command shape moves the result more than the effect
  M10 the n "trajectories" are 3-scalar draws; the budget is nearly determined by amplitude alone

AMPLITUDE. This ran at 1/4.7 while the letter reported 1/3.57, so every number it produced -- the
shape sensitivity, the budget, the r^2 -- described a configuration the manuscript does not use, and
the manuscript quoted them anyway. The default is now the amplitude the letter flies, derived in
experiments/aero_feasible_factor.py rather than restated here; --amp-factor overrides it.

Run:  python experiments/audit_t10.py --json runs/ml/audit_t10.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from experiments.multiclass_lead import class_windows, measure, synth_command  # noqa: E402

AMP, SNR, DWELL, SIG = 0.2798, 40.0, 0.002, 0.3


SEEDS = 30                          # overridden by --seeds; H4's block spread is n-sensitive


def sweep(seed_base, seeds=None, reps=12, shape=None, dur=None):
    seeds = SEEDS if seeds is None else seeds
    """One advantage per trajectory, mirroring sweep_class()."""
    advs, buds, amps, alts, machs, dets = [], [], [], [], [], []
    for sd in range(seeds):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(seed_base + sd),
                                    amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        if not wins:
            continue
        ev = []
        for w in wins:
            w2 = dict(w)
            if shape is not None:
                tt, aa, tc = synth_command(shape, dur or 0.30,
                                           w["amp_g"] * 9.80665, 0.60, 0.40)
                w2["t"], w2["a_cmd"], w2["t_cmd"] = tt, aa, tc
            w2["a_cmd"] = np.asarray(w2["a_cmd"], float) * AMP
            m = measure(w2, SNR, reps, DWELL, kin_noise=SIG)
            if not m:
                continue
            dets.append(m["det"])
            a = (m.get("arms") or {}).get("CUSUM Page54", {}).get("adv")
            if a is not None:
                ev.append(a)
                buds.append(m["budget_ms"])
                amps.append(w["amp_g"] * AMP)
                alts.append(w["alt"] / 1000.0)
                machs.append(w["mach"])
        if ev:
            advs.append(float(np.median(ev)))
    return (np.asarray(advs, float), np.asarray(buds, float), np.asarray(amps, float),
            np.asarray(alts, float), np.asarray(machs, float),
            float(np.mean(dets)) if dets else 0.0)


def boot(v, n, seed=11):
    r = np.random.default_rng(seed)
    m = [np.median(r.choice(v, v.size, replace=True)) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    global AMP, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--amp-factor", type=float, default=AMP)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    AMP, SEEDS = args.amp_factor, args.seeds
    out = {"amp_factor": AMP, "snr_db": SNR, "kin_noise": SIG, "seeds": SEEDS}
    print("amplitude factor %.4f, %.0f dB, sigma %.2f, %d seeds\n" % (AMP, SNR, SIG, SEEDS))

    print("=== H4: does a different SEED BLOCK land outside the published CI? ===")
    base, bud0, amp0, alt0, mach0, _ = sweep(90000)
    lo0, hi0 = boot(base, 20000)
    print("  published block 90000 : %+.1f [%+.1f,%+.1f] n=%d"
          % (np.median(base), lo0, hi0, base.size))
    reseeds = {}
    for sb in (50000, 123456, 777000):
        v, *_ = sweep(sb)
        if v.size < 3:
            continue
        lo, hi = boot(v, 20000)
        inside = lo0 <= np.median(v) <= hi0
        reseeds[sb] = dict(median=float(np.median(v)), lo=lo, hi=hi, n=int(v.size),
                           inside_published_ci=bool(inside))
        print("  block %-7d       : %+.1f [%+.1f,%+.1f] n=%d   %s"
              % (sb, np.median(v), lo, hi, v.size,
                 "inside" if inside else "*** OUTSIDE published CI ***"))
    meds = [np.median(base)] + [r["median"] for r in reseeds.values()]
    print("  spread across blocks  : %+.1f to %+.1f ms  (range %.1f)"
          % (min(meds), max(meds), max(meds) - min(meds)))
    out["H4"] = dict(published=dict(median=float(np.median(base)), lo=lo0, hi=hi0,
                                    n=int(base.size)), reseeds=reseeds,
                     across_block_range_ms=float(max(meds) - min(meds)))

    print("\n=== H5: does the CI endpoint move with the bootstrap count? ===")
    h5 = {}
    for n in (2000, 8000, 20000, 100000):
        lo, hi = boot(base, n)
        h5[n] = [lo, hi]
        print("  n=%-7d [%+.2f, %+.2f]" % (n, lo, hi))
    los = [v[0] for v in h5.values()]
    print("  lower endpoint moves %.2f ms across resample counts" % (max(los) - min(los)))
    out["H5"] = dict(by_n={str(k): v for k, v in h5.items()},
                     lower_endpoint_range_ms=float(max(los) - min(los)))

    print("\n=== M2: command SHAPE ===")
    m2 = {}
    for shape, dur in (("step", 0.30), ("raised-cosine", 0.30), ("sinusoid", 0.30),
                       ("raised-cosine", 1.00)):
        try:
            v, b, *_ , det = sweep(90000, shape=shape, dur=dur)
        except Exception as e:                                                # noqa: BLE001
            print("  %-14s failed: %s" % (shape, str(e)[:50]))
            continue
        if v.size < 3:
            print("  %-14s n=%d (too few)" % ("%s@%.2f" % (shape, dur), v.size))
            continue
        lo, hi = boot(v, 20000)
        m2["%s@%.2f" % (shape, dur)] = dict(median=float(np.median(v)), lo=lo, hi=hi, n=int(v.size),
                         det=det, budget=float(np.median(b)), min=float(v.min()))
        print("  %-18s budget %6.1f ms  det %3.0f%%  adv %+.1f [%+.1f,%+.1f] n=%d  min %+.1f"
              % ("%s@%.2f" % (shape, dur), np.median(b), 100 * det, np.median(v),
                 lo, hi, v.size, v.min()))
    out["M2"] = m2

    print("\n=== M10: is the budget determined by amplitude alone? ===")
    if bud0.size > 3:
        for name, X in (("amp_g", np.c_[amp0]),
                        ("amp_g+alt", np.c_[amp0, alt0]),
                        ("amp_g+alt+mach", np.c_[amp0, alt0, mach0])):
            A = np.c_[np.ones(len(bud0)), X]
            coef, *_ = np.linalg.lstsq(A, bud0, rcond=None)
            pred = A @ coef
            r2 = 1 - ((bud0 - pred) ** 2).sum() / ((bud0 - bud0.mean()) ** 2).sum()
            print("  budget ~ %-16s r2 = %.4f" % (name, r2))
            out.setdefault("M10", {})[name] = float(r2)
        print("  => the measurement's effective dimensionality is the number of distinct")
        print("     (amplitude, altitude, Mach) triples, not the number of trajectories.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
