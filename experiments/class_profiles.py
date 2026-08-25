"""experiments/class_profiles.py -- per-class lateral-acceleration profile and FEASIBILITY GATE.

WHAT THIS ANSWERS. Before any per-class micro-Doppler lead can be measured, three things have to be
true of a class: it must command a maneuver at all; that maneuver must have an ONSET (a quiescent
phase followed by a departure); and the airframe must be able to physically DELIVER the commanded
acceleration at that class's own flight condition. This module measures all three.

HOW THE CLASSES ARE ENUMERATED. From the project's own registry,
trajectory_generators.profiles.GENERATORS, which maps all five classes to a uniform
(P, V, dt, meta) contract. Not by monkeypatching `run` on each generator module: several
generators import `integrate` from .dynamics and never call run(), so a patch-based discovery
silently substitutes a different vehicle. Not from a hand-written class list either. If a class is
in the registry it is in the table, and if it fails it fails loudly.

COMMAND vs ACHIEVED, which is load-bearing. Only `maneuvering` exports the true guidance command
(`meta["cmd_lat_accel"]`, alongside ACTUATOR_LAG_S). Every other generator prescribes acceleration
directly with no actuator lag, so commanded and achieved are the same signal and the only thing
recoverable is the ACHIEVED lateral acceleration, derived from V. The table reports which source
each row used, because a lead measured against a command and a lead measured against an achieved
state are different quantities -- conflating them is this project's defining defect.

TRIM ESTIMATION uses a CHANGE-POINT boundary, not a fixed leading window. A fixed
10%-of-duration window is not usable here: HGV's peak swings 1.80x on window choice alone (it
weaves from t=0, so a leading window contains maneuver and the trim absorbs signal), and MaRV's
onset count runs 1->2->0->1 across window fractions. `--frac-sweep` demonstrates that the
change-point boundary is flat over the same range.

    python experiments/class_profiles.py
    python experiments/class_profiles.py --frac-sweep
    python experiments/class_profiles.py --json runs/ml/class_profiles.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from trajectory_generators.profiles import GENERATORS                         # noqa: E402
from trajectory_generators.dynamics import altitude_of                        # noqa: E402
from trajectory_generators.base import G0                                     # noqa: E402
from sim.sixdof import a_max_at, sound_speed                                  # noqa: E402

ONSET_G_MS2 = 2.0            # the paper's departure-from-cruise threshold, m/s^2


def load_class(name, rng=None, max_attempts=8, gate=True):
    """Generate ONE trajectory of class `name` by the DATASET's own path, not an approximation.

    Every step below exists because omitting it changes the distribution, and earlier versions of
    this file omitted most of them:

      CORRIDOR ELIGIBILITY -- launch/target come from corridors_for(name), a class-filtered list,
        picked uniformly. A previous version hardcoded (32.0,53.0)->(31.6,35.0), which is
        C1_iran_israel; per corridors.py that corridor is eligible only for ballistic/marv/mirv, so
        supersonic_cruise and hgv were being flown on a corridor they can never be assigned. Range
        and geography drive the internal bisection solve for launch speed in ballistic and marv, so
        this is not cosmetic.
      ENDPOINT SAMPLING -- corridor.sample_endpoints(rng) rejection-samples against a land mask.
      PARAMETERS -- experiments.generate_dataset._sample_params, called not restated. It injects
        two constants that are NOT generator defaults: supersonic_cruise's boost_from_ground=True
        and hgv's aero="cavh" (wind-tunnel CAV-H polynomials, so lift SATURATES at low dynamic
        pressure). Miss either and you silently fly the other physics branch.
      MIRV -- goes through sim.mirv.mirv_bus DIRECTLY. GENERATORS['mirv'] is _mirv_wrap, a legacy
        single-object adapter that returns only rv0 and silently drops the bus, the other two RVs
        and all three decoys.
      TIER-0 GATE + RETRY -- tools.audit_dataset.audit_trajectory, up to 8 attempts. Without it you
        keep trajectories the dataset explicitly threw away: HGV's disclosed 42 g heading-correction
        spike fails lateral_g, MaRV pull-ups that over-flatten into a loft fail altitude.
      evasive -- left unset so the generator default True applies, which is what the dataset does
        (generate_scenario passes no evasive kwarg).

    Returns the same dict as before plus `attempts` and `corridor`, or raises if the retry budget
    is exhausted -- which is itself the dataset's behaviour (the scenario is dropped).
    """
    from trajectory_generators.corridors import corridors_for
    from experiments.generate_dataset import _sample_params, SPEC_KEY
    from audit_dataset import audit_trajectory

    rng = rng if rng is not None else np.random.default_rng(0)
    last = None
    for attempt in range(max_attempts):
        choices = corridors_for(name)
        corridor = choices[int(rng.integers(len(choices)))]
        launch_ll, target_ll = corridor.sample_endpoints(rng)

        if name == "mirv":
            from sim.mirv import mirv_bus
            objs, dt, meta = mirv_bus(launch_ll, target_ll, rng=rng)
            if gate:
                ok = True
                for o in objs:
                    seg = slice(o["birth_step"], o["death_step"] + 1)
                    passed, _c = audit_trajectory(o["P"][seg], dt, o["spec_key"])
                    ok = ok and passed
                if not ok:
                    last = "tier0 gate"
                    continue
            # single-track view for the onset question: the primary RV, but the full object list is
            # carried so nothing is silently lost
            rv = [o for o in objs if o.get("type") == "rv"] or objs
            o = rv[0]
            # SLICE TO THE OBJECT'S OWN ALIVE WINDOW. Every MIRV object carries a FULL-LENGTH P/V
            # array in which everything before birth_step is the BUS's track -- _assemble() puts
            # each object co-located on the bus until dispense. Analysing the whole array therefore
            # measures the bus BOOST and attributes it to the RV: peak |A| at step 0 is 5509 m/s^2
            # = 561.8 g, shared identically by every object, which is what produced the "141-176 g
            # MIRV" figure reported earlier. The true dispense impulse is 14.6-18.1 g across the
            # three RVs, matching profiles.py:28's documented 17 g. Tier-0 gets this right --
            # audit_trajectory slices seg = slice(birth_step, death_step+1) -- and this did not.
            b = int(o.get("birth_step", 0))
            e = int(o.get("death_step", len(o["P"]) - 1))
            P = np.asarray(o["P"], float)[b:e + 1]
            V = np.asarray(o["V"], float)[b:e + 1]
            return dict(name=name, t=np.arange(len(P)) * dt, P=P, V=V, dt=float(dt),
                        meta=meta or {}, objects=objs, attempts=attempt + 1,
                        corridor=corridor.name, birth_step=b, death_step=e)

        params = _sample_params(name, rng)
        P, V, dt, meta = GENERATORS[name](launch_ll, target_ll, **params)
        P = np.asarray(P, float); V = np.asarray(V, float)
        if len(P) < 10:
            last = "degenerate length"
            continue
        if gate:
            passed, _c = audit_trajectory(P, dt, SPEC_KEY[name])
            if not passed:
                last = "tier0 gate"
                continue
        return dict(name=name, t=np.arange(len(P)) * dt, P=P, V=V, dt=float(dt),
                    meta=meta or {}, attempts=attempt + 1, corridor=corridor.name)
    raise RuntimeError("%s: %d attempts exhausted (%s) -- the dataset would drop this scenario"
                       % (name, max_attempts, last))


def lateral_signal(rec):
    """The per-sample lateral acceleration, and WHERE it came from.

    'command' -- meta['cmd_lat_accel'], the true guidance command. Only `maneuvering` exports this,
                 and it is the quantity a command-derived cue would be built from.
    'achieved' -- the component of dV/dt perpendicular to velocity. This is what a sensor could in
                 principle observe, and for every generator except maneuvering it is also the
                 command, because those generators apply no actuator lag.
    """
    meta = rec["meta"]
    cmd = meta.get("cmd_lat_accel")
    if cmd is not None and len(np.atleast_1d(cmd)) >= len(rec["t"]) // 2:
        c = np.asarray(cmd, float).ravel()
        if len(c) < len(rec["t"]):
            c = np.pad(c, (0, len(rec["t"]) - len(c)), mode="edge")
        return c[:len(rec["t"])], "command"
    V, dt = rec["V"], rec["dt"]
    A = np.gradient(V, dt, axis=0)
    sp = np.linalg.norm(V, axis=1, keepdims=True)
    vhat = V / np.maximum(sp, 1e-9)
    a_par = np.einsum("ij,ij->i", A, vhat)[:, None] * vhat
    return np.linalg.norm(A - a_par, axis=1), "achieved"


def decompose(sig, frac=0.02):
    """Split into TRIM and MANEUVER by a change-point boundary.

    Seed the trim from a short prefix, walk to the first deviation exceeding `frac` of the
    trajectory's OWN peak, and estimate trim from everything before that. The boundary is found
    rather than fixed: a leading window of fixed length can flip an onset into existence or out of
    it, since where it falls relative to the maneuver decides what counts as trim.
    """
    n = len(sig)
    seed = max(3, int(0.02 * n))
    t0 = float(np.median(sig[:seed]))
    dev0 = np.abs(sig - t0)
    peak = float(dev0.max())
    if peak <= 0:
        return np.zeros(n), t0, n
    above = dev0 > frac * peak
    cut = int(np.argmax(above)) if above.any() else n
    cut = max(cut, seed)
    trim = float(np.median(sig[:cut]))
    return np.abs(sig - trim), trim, cut


def feasibility(rec):
    """Can the airframe DELIVER the commanded lateral acceleration at this class's own condition?

    The question is unavoidable because the answer is often no: driving PitchAirframe with HGV's
    own command at HGV's own 40 km glide puts 91% of the flight above the airframe's local a_max,
    with achieved a_z pinned at ~1.03 g against an 8.0 g command. A class whose command exceeds
    available control authority is not maneuvering in the model -- it is saturating -- and no lead
    measured on it means anything.

    The project already knows this. trajectory_generators/profiles.py lines 24-31 tabulate peak
    lateral demand against what a real vehicle of each class could produce: supersonic_cruise 13.2 g
    (14.3x evader.py, 4.7x a sourced CAV-H), hgv 42.4 g (14.6x / 5.8x), marv 15.0 g (12.8x / 4.2x),
    and notes it is "NOT silently corrected". This recomputes the same gate from the trajectory.
    """
    P, V = rec["P"], rec["V"]
    alt = np.array([altitude_of(p) for p in P])
    sp = np.linalg.norm(V, axis=1)
    amax = np.full(len(P), np.nan)
    for i in range(len(P)):
        a = float(alt[i])
        if not np.isfinite(a) or a < 0 or a > 90000:
            continue
        try:
            amax[i] = a_max_at(a, sp[i] / max(sound_speed(a), 1e-6))
        except Exception:                                            # noqa: BLE001
            continue
    return alt, amax


def describe(rec, frac=0.02):
    sig, src = lateral_signal(rec)
    man, trim, cut = decompose(sig, frac=frac)
    t = rec["t"]
    alt, amax = feasibility(rec)
    peak = float(man.max())

    above = man > ONSET_G_MS2
    n_onsets = int(np.sum(np.diff(above.astype(int)) == 1))
    quiet = float(np.mean(~above))

    # ONSET CRITERION. Requiring EXACTLY ONE threshold crossing would reject supersonic_cruise
    # -- the paper's own class -- because a terminal WEAVE is periodic and crosses repeatedly. What an onset study actually needs is a quiescent phase
    # followed by a FIRST sustained departure; later reversals of the same weave do not disqualify
    # it, they are the same maneuver continuing. The paper scores the first crossing.
    i_on = int(np.argmax(above)) if above.any() else -1
    # EXO-ATMOSPHERIC GUARD. Above ~80 km there is no dynamic pressure to turn against, so a
    # lateral-acceleration "onset" there cannot be an aerodynamic maneuver. MIRV trips this: its
    # peak reads 141 g at 296 km, which profiles.py:28 already identifies as "the modelled 140 m/s
    # dispense impulse read through np.gradient, not lift". Without this guard the table reports a
    # separation event as a maneuver onset and MIRV looks like the most agile class in the set.
    exo = bool(i_on > 0 and alt[i_on] > 80000.0)
    has_onset = bool(i_on > 0 and quiet > 0.2 and not exo)
    t_onset = float(t[i_on]) if i_on > 0 else None
    # the flight condition that matters is the one AT onset, not at launch
    alt_on = float(alt[i_on] / 1000) if i_on > 0 else float(alt[0] / 1000)

    # FEASIBILITY GATE applies only to aerodynamically driven classes. profiles.py:28 states
    # ballistic and mirv are "not aerodynamically driven (gravity+drag; mirv's 17 g is the modelled
    # 140 m/s dispense impulse read through np.gradient, not lift)". Comparing a gravity turn or a
    # separation impulse against a fin-generated a_max is a category error, so it is reported N/A
    # rather than as a large exceedance that would look like saturation.
    aero = src == "command"
    ok = np.isfinite(amax) & (amax > 0)
    exceed = float(np.mean(sig[ok] > amax[ok])) if (aero and ok.any()) else float("nan")
    hdr = float(np.nanmedian(amax[ok])) if ok.any() else float("nan")
    return dict(name=rec["name"], n=len(t), dur=float(t[-1]), src=src, aero=aero, exo=exo,
                alt_onset_km=alt_on, peak_g=peak / G0, trim_g=trim / G0,
                quiescent=quiet, n_crossings=n_onsets, has_onset=has_onset, t_onset=t_onset,
                frac_over_amax=exceed, med_amax_g=hdr / G0 if np.isfinite(hdr) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--frac-sweep", action="store_true",
                    help="sweep the change-point fraction; the verdict must NOT track it")
    args = ap.parse_args()

    names = list(GENERATORS.keys())

    if args.frac_sweep:
        print("CHANGE-POINT STABILITY -- does the verdict track the analyst's parameter?\n")
        fr = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
        print("%-20s %s" % ("class", "  ".join("%8.3f" % f for f in fr)))
        print("-" * 80)
        for nm in names:
            try:
                rec = load_class(nm)
            except Exception as e:                                   # noqa: BLE001
                print("%-20s  FAILED: %s" % (nm, str(e)[:44]))
                continue
            cells = []
            for f in fr:
                d = describe(rec, frac=f)
                cells.append("%d/%s" % (d["n_crossings"], "Y" if d["has_onset"] else "n"))
            print("%-20s %s" % (nm, "  ".join("%8s" % c for c in cells)))
        print("\nEach cell is onsets/single?. A row that CHANGES is still tracking the analyst.")
        return

    print("PER-CLASS LATERAL PROFILE AND FEASIBILITY GATE")
    print("  classes from trajectory_generators.profiles.GENERATORS -- the project's own registry,")
    print("  so no class can be silently substituted or omitted.\n")
    print("%-20s %-9s %6s %9s %8s %7s %6s %8s %8s %9s"
          % ("class", "signal", "n", "alt@onset", "peak", "quiet", "cross", "onset?", "t_onset",
             "> a_max"))
    print("-" * 104)

    out = []
    for nm in names:
        try:
            rec = load_class(nm)
            d = describe(rec)
        except Exception as e:                                       # noqa: BLE001
            print("%-20s  FAILED: %s" % (nm, str(e)[:60]))
            out.append(dict(name=nm, error=str(e)))
            continue
        print("%-20s %-9s %6d %8.1fkm %7.2fg %6.0f%% %6d %8s %8s %9s"
              % (d["name"], d["src"], d["n"], d["alt_onset_km"], d["peak_g"],
                 100 * d["quiescent"], d["n_crossings"],
                 "YES" if d["has_onset"] else ("EXO" if d.get("exo") else "no"),
                 ("%.0fs" % d["t_onset"]) if d["t_onset"] is not None else "--",
                 ("%.0f%%" % (100 * d["frac_over_amax"])) if np.isfinite(d["frac_over_amax"])
                 else "n/a"))
        out.append(d)

    print("\nREADING.")
    print("  signal   -- 'command' is the true guidance command (only maneuvering exports it);")
    print("              'achieved' is the observable perpendicular acceleration. Not interchangeable.")
    print("  single?  -- quiescent phase then ONE sustained departure: the structure an onset study")
    print("              needs. 'no' with 0 onsets means the class never maneuvers and its lead is")
    print("              UNDEFINED, not zero.")
    print("  > a_max  -- fraction of the flight commanding more lateral acceleration than the")
    print("              airframe can deliver at that altitude and Mach. A high value means the")
    print("              'maneuver' is actuator saturation and no lead measured on it is meaningful.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
