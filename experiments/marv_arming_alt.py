"""experiments/marv_arming_alt.py -- MaRV onset advantage versus pull-up arming altitude.

The paper excludes MaRV, which maneuvers at 11.95 g, because at its 52 km arming altitude every
event saturates the airframe and the race measures control authority. This script measures how
that result depends on the arming altitude. Generator properties that bear on it
(trajectory_generators/profiles.py):

  * the manoeuvre band is gated on `alt < pullup_trigger_alt` with no dynamic-pressure term;
  * experiments/generate_dataset._sample_params samples pullup_trigger_alt from 50-62 km;
  * at ~57 km (qbar ~ 2.3 kPa) the airframe can pull ~0.11 g (CAV-H) or ~0.04 g (evader surrogate)
    against a commanded ~18 g, a ~139x to ~384x over-command;
  * a 12-17 g aero jink first becomes feasible around 14-23 km;
  * the project's flight-profile specification arms the pull-up at ~40 km;
  * marv_arc defaults to pullup_trigger_alt = 25 km, and the dataset sampler overrides it upward.

MaRV is the one class whose command is a bang-bang pull-up plus weave, and the shape sweep shows
the cue responds to fast transients and fades on slow ones. The script reruns the MaRV cell with
the pull-up armed at 56 km (dataset sampling), 40 km (specification), 25 km (generator default)
and 18 km (a flyable band). A cell converts when n >= 3 and detection >= 50%.

    python experiments/marv_arming_alt.py --json runs/ml/marv_arming.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SNR, REPS, DWELL, SIGMA, SEEDS = 40.0, 12, 0.002, 0.3, 30
# dataset sampling, flight-profile specification, generator default, and one flyable band
ALTS_KM = (56.0, 40.0, 25.0, 18.0)


def run_cell(alt_km):
    """Run one n=30 MaRV cell with the pull-up armed at `alt_km` (km) and all other parameters as
    sampled. Returns a dict of per-cell statistics."""
    import numpy as np
    from scipy.stats import wilcoxon
    import experiments.multiclass_lead as ml
    import experiments.generate_dataset as gd
    from experiments.multiclass_lead import class_windows, measure

    # Patch only the arming altitude, in the sampler the pipeline calls. pullup_g, weave_g,
    # corridor and beta keep the values the dataset draws.
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

    # multiclass_lead.SHAPES declares MaRV as the raised-cosine pull-up the generator commands
    # (marv.py env = sin(pi*m) over maneuver_dur_s).
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
    print("MaRV arming-altitude sweep\n")
    print("  airframe lateral-acceleration limit at each altitude (the dataset commands ~18 g):")
    for a in ALTS_KM:
        print("     %4.0f km   a_max = %7.3f g" % (a, a_max_at(a * 1000.0, 3.0) / G0))
    print("\n  dataset sampling: 50-62 km   project spec: ~40 km   generator default: 25 km\n")

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
        print("A converting cell at or below the 40 km specification means the MaRV exclusion")
        print("depends on the arming-altitude sampling.")
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(snr_db=SNR, sigma=SIGMA, seeds=SEEDS, reps=REPS,
                       alts_km=list(ALTS_KM), rows=rows), open(args.json, "w"),
                  indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
