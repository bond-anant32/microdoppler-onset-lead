"""experiments/marv_arming_alt.py -- is MaRV's exclusion physics, or a dataset sampling bug?

THE CLAIM UNDER TEST. Sec. IV says: "MaRV maneuvers (11.95 g) but at its 52 km arming altitude every
event saturates the airframe, measuring control authority, not onset." That is presented as a
PHYSICAL property of the class. It is not. From trajectory_generators/profiles.py's own disclosure,
marked "THIS IS THE WORST GAP IN THE PROJECT":

  * the manoeuvre band is gated on `alt < pullup_trigger_alt` with NO dynamic-pressure term;
  * experiments/generate_dataset._sample_params samples pullup_trigger_alt from 50-62 km;
  * at ~57 km (qbar ~ 2.3 kPa) the airframe can pull ~0.11 g (CAV-H) or ~0.04 g (evader surrogate),
    against a commanded ~18 g -- i.e. ~139x to ~384x over-command;
  * "A 12-17 g aero jink first becomes feasible around 14-23 km";
  * study/FLIGHT_PROFILE_SPEC.md line 27 specifies arming at ~40 km;
  * and marv_arc's OWN DEFAULT is pullup_trigger_alt = 25 km. The dataset overrides it upward.

So the class is excluded at an arming altitude that contradicts the project's spec AND the
generator's own default. "Every event saturates" is then a statement about the sampling, not about
MaRV. If MaRV converts when armed where the spec says, the letter's "one class of five" is wrong
and its scope claim has to be rewritten.

MaRV also matters more than the count suggests: it is the one class whose command is a BANG-BANG
pull-up plus weave, and the shape sweep shows the cue lives on fast transients and dies on slow
ones. If any class should convert, it is this one.

    python experiments/marv_arming_alt.py --json runs/ml/marv_arming.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SNR, REPS, DWELL, SIGMA, SEEDS = 40.0, 12, 0.002, 0.3, 30
# shipped sampling, the project's own spec, the generator's own default, and one flyable band
ALTS_KM = (56.0, 40.0, 25.0, 18.0)


def run_cell(alt_km):
    """One n=30 MaRV cell with the pull-up armed at `alt_km`, everything else as shipped."""
    import numpy as np
    from scipy.stats import wilcoxon
    import experiments.multiclass_lead as ml
    import experiments.generate_dataset as gd
    from experiments.multiclass_lead import class_windows, measure

    # Patch ONLY the arming altitude, in the sampler the pipeline actually calls. Everything else --
    # pullup_g, weave_g, corridor, beta -- stays exactly as the dataset draws it.
    _orig = gd._sample_params

    def _patched(missile_type, rng):
        p = _orig(missile_type, rng)
        if missile_type == "marv" and "pullup_trigger_alt" in p:
            p["pullup_trigger_alt"] = float(alt_km * 1000.0)
        return p

    gd._sample_params = _patched
    ml._sample_params = _patched
    import experiments.class_profiles as cp
    if hasattr(cp, "_sample_params"):
        cp._sample_params = _patched

    # MaRV is not in SHAPES with a step; the generator commands a raised-cosine pull-up (marv.py
    # env = sin(pi*m) over maneuver_dur_s), which is what multiclass_lead already declares.
    advs, mus, kins, dets, buds, gs = [], [], [], [], [], []
    n_win = n_seed = 0
    for sd in range(SEEDS):
        try:
            wins, d = class_windows("marv", rng=np.random.default_rng(90000 + sd))
        except Exception:                                                     # noqa: BLE001
            continue
        n_seed += 1
        if not wins:
            continue
        n_win += 1
        ea, em, ek = [], [], []
        for w in wins:
            gs.append(w["amp_g"])
            try:
                m = measure(w, SNR, REPS, DWELL, kin_noise=SIGMA)
            except Exception:                                                 # noqa: BLE001
                continue
            if not m:
                continue
            dets.append(m["det"]); buds.append(m["budget_ms"])
            if m.get("muD") is not None:
                em.append(m["muD"])
            r = (m.get("arms") or {}).get("CUSUM Page54") or {}
            if r.get("lead") is not None:
                ek.append(r["lead"])
            if r.get("adv") is not None:
                ea.append(r["adv"])
        if ea:
            advs.append(float(np.median(ea)))
        if em:
            mus.append(float(np.median(em)))
        if ek:
            kins.append(float(np.median(ek)))

    a = np.asarray(advs, float)
    med = lambda v: float(np.median(v)) if len(v) else None                   # noqa: E731
    return dict(arming_alt_km=float(alt_km), n_seeds=n_seed, n_with_event=n_win,
                n=int(a.size), adv_median=med(a),
                worst=float(a.min()) if a.size else None, n_pos=int((a > 0).sum()),
                muD_lead_ms=med(mus), cusum_lead_ms=med(kins), budget_ms=med(buds),
                mean_g=float(np.mean(gs)) if gs else None,
                det=float(np.mean(dets)) if dets else None,
                p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--procs", type=int, default=max(1, min(6, cpu_count() - 2)))
    args = ap.parse_args()

    from sim.sixdof import a_max_at, G0
    print("MaRV ARMING ALTITUDE -- is the exclusion physics or sampling?\n")
    print("  what the airframe can pull, vs the ~18 g the dataset commands:")
    for a in ALTS_KM:
        print("     %4.0f km   a_max = %7.3f g" % (a, a_max_at(a * 1000.0, 3.0) / G0))
    print("\n  shipped sampling: 50-62 km   project spec: ~40 km   generator default: 25 km\n")

    with Pool(args.procs, maxtasksperchild=1) as pool:
        rows = pool.map(run_cell, ALTS_KM, chunksize=1)

    print("%9s %8s %8s %5s %11s %9s %6s %9s %7s"
          % ("arm km", "seeds", "w/event", "n", "advantage", "worst", "pos", "muD", "det"))
    print("-" * 84)
    for r in rows:
        f = lambda v, p="%+.2f": (p % v) if v is not None else "--"           # noqa: E731
        print("%9.0f %8d %8d %5d %11s %9s %3d/%-2d %9s %6.0f%%"
              % (r["arming_alt_km"], r["n_seeds"], r["n_with_event"], r["n"],
                 f(r["adv_median"]), f(r["worst"], "%+.1f"), r["n_pos"], r["n"],
                 f(r["muD_lead_ms"]), 100 * (r["det"] or 0)))

    conv = [r for r in rows if r["n"] >= 3 and (r["det"] or 0) >= 0.5]
    print("\nCells that CONVERT (n>=3, detection >=50%%): %s"
          % (", ".join("%.0f km" % r["arming_alt_km"] for r in conv) or "none"))
    if conv:
        print("If any of these is at or below the project's own 40 km spec, then 'one class of")
        print("five' is a property of the SAMPLING, not of the class, and Sec. IV must be rewritten.")
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(snr_db=SNR, sigma=SIGMA, seeds=SEEDS, reps=REPS,
                       alts_km=list(ALTS_KM), rows=rows), open(args.json, "w"),
                  indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
