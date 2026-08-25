"""experiments/true_shape_test.py -- the command shape the GENERATOR actually commands.

THE ISSUE. experiments/multiclass_lead.SHAPES maps

    "supersonic_cruise": ("step", 0.0)

while trajectory_generators/maneuvering.py commands

    a_lat_command(t) = weave_lat_accel * sin(2*pi*weave_cycles*m),  m = (t-weave_start_t)/weave_dur

-- a SINUSOID, with weave_dur = clip((1-weave_start_frac)*duration, 20, 60) s and weave_cycles
sampled at 3.5-4.9 by generate_dataset. The half-period at onset is therefore several seconds, and
the headline is measured under a mathematical STEP: infinite initial slope, the most favorable
possible input for a statistic that goes as the SQUARE of fin rate.

The letter discloses shape as an authored axis and reports step/raised-cosine/sinusoid at
+17.0/+44.5/+42.5 ms. This script measures the operating point under the generator's OWN shape and
period, so the headline can be read at the input the trajectory set specifies rather than at a
convenient one.

    A  step                   -- CONTROL, must reproduce +17.0 ms
    B  sinusoid, per-seed period taken from that seed's own weave_cycles and weave_dur
    C  sinusoid at a fixed 36 s period, for continuity with audit_t10's M2 row

WHAT WOULD FALSIFY THE HEADLINE. If B collapses or reverses, the +17.0 ms is an artifact of a step
the generator never commands, and the letter must be re-anchored on B.

    python experiments/true_shape_test.py --json runs/ml/true_shape.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from scipy.stats import wilcoxon                                             # noqa: E402

AMP, SNR, REPS, DWELL, SIGMA, SEEDS = 0.2798, 40.0, 12, 0.002, 0.3, 30


def weave_period_s(seed):
    """The seed's OWN weave half-period, read from the generator's parameters -- not restated."""
    from experiments.generate_dataset import _sample_params
    p = _sample_params("supersonic_cruise", np.random.default_rng(90000 + seed))
    dur = float(p.get("duration_s", 400.0))
    frac = float(p.get("weave_start_frac", 0.6))
    cycles = float(p.get("weave_cycles", 4.0))
    weave_dur = float(np.clip((1.0 - frac) * dur, 20.0, 60.0))
    # synth_command's `dur_s` is the FULL period of one sine; the generator packs `cycles` of them
    # into weave_dur, so one period is weave_dur/cycles.
    return weave_dur / max(cycles, 1e-9)


def run(mode):
    import experiments.multiclass_lead as ml
    from experiments.multiclass_lead import class_windows, measure

    advs, mus, kins, dets, buds = [], [], [], [], []
    for sd in range(SEEDS):
        if mode == "step":
            ml.SHAPES = dict(ml.SHAPES); ml.SHAPES["supersonic_cruise"] = ("step", 0.0)
        elif mode == "true":
            ml.SHAPES = dict(ml.SHAPES)
            ml.SHAPES["supersonic_cruise"] = ("sinusoid", weave_period_s(sd))
        else:                                            # fixed 36 s, audit_t10's M2 row
            ml.SHAPES = dict(ml.SHAPES); ml.SHAPES["supersonic_cruise"] = ("sinusoid", 36.0)
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        if not wins:
            continue
        ev_a, ev_m, ev_k = [], [], []
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
            m = measure(w2, SNR, REPS, DWELL, kin_noise=SIGMA)
            if not m:
                continue
            dets.append(m["det"]); buds.append(m["budget_ms"])
            if m.get("muD") is not None:
                ev_m.append(m["muD"])
            r = (m.get("arms") or {}).get("CUSUM Page54") or {}
            if r.get("lead") is not None:
                ev_k.append(r["lead"])
            if r.get("adv") is not None:
                ev_a.append(r["adv"])
        if ev_a:
            advs.append(float(np.median(ev_a)))
        if ev_m:
            mus.append(float(np.median(ev_m)))
        if ev_k:
            kins.append(float(np.median(ev_k)))

    a = np.asarray(advs, float)
    med = lambda v: float(np.median(v)) if len(v) else None                   # noqa: E731
    return dict(mode=mode, n=int(a.size), adv_median=med(a),
                worst=float(a.min()) if a.size else None, n_pos=int((a > 0).sum()),
                muD_lead_ms=med(mus), cusum_lead_ms=med(kins), budget_ms=med(buds),
                det=float(np.mean(dets)) if dets else None,
                p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None,
                adv=[float(x) for x in a])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    per = [weave_period_s(s) for s in range(SEEDS)]
    print("TRUE COMMAND SHAPE -- the generator commands a SINUSOID, not a step")
    print("  per-seed weave period from the generator's own parameters: %.1f-%.1f s (median %.1f)\n"
          % (min(per), max(per), float(np.median(per))))

    rows = [run("step"), run("true"), run("sin36")]
    print("%-28s %4s %11s %9s %6s %9s %9s %7s"
          % ("command shape", "n", "advantage", "worst", "pos", "muD", "CUSUM", "det"))
    print("-" * 88)
    for r in rows:
        lab = {"step": "A step (shipped headline)",
               "true": "B generator's own sinusoid",
               "sin36": "C sinusoid, fixed 36 s"}[r["mode"]]
        f = lambda v, p="%+.2f": (p % v) if v is not None else "--"           # noqa: E731
        print("%-28s %4d %11s %9s %3d/%-2d %9s %9s %6.0f%%"
              % (lab, r["n"], f(r["adv_median"]), f(r["worst"], "%+.1f"), r["n_pos"], r["n"],
                 f(r["muD_lead_ms"]), f(r["cusum_lead_ms"]), 100 * (r["det"] or 0)))

    ctrl = rows[0]["adv_median"]
    print("\nCONTROL: the step row must reproduce the shipped +17.0 ms -> %s"
          % ("OK" if ctrl is not None and abs(ctrl - 17.0) < 1e-6 else "*** MISMATCH ***"))
    t = rows[1]["adv_median"]
    if ctrl is not None and t is not None:
        print("Under the shape the generator ACTUALLY commands the advantage is %+.2f ms (%+.2f vs "
              "the step)." % (t, t - ctrl))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(sigma=SIGMA, snr_db=SNR, seeds=SEEDS, reps=REPS, amp_factor=AMP,
                       weave_period_s=per, rows=rows), open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
