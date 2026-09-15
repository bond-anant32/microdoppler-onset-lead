"""experiments/class_profiles.py -- per-class lateral-acceleration profile and aerodynamic feasibility.

For each trajectory class this measures whether the class commands a maneuver, whether that maneuver
has an onset (a quiescent phase followed by a departure above ONSET_G_MS2), and whether the airframe
can deliver the commanded acceleration at the class's own flight condition.

Classes are enumerated from the registry trajectory_generators.profiles.GENERATORS, which maps all
five classes to a uniform (P, V, dt, meta) contract. Several generators import `integrate` from
.dynamics and never call run(), so patching `run` on each generator module would not reach them.
A class that fails to generate is reported as FAILED in the table.

Only `maneuvering` (supersonic_cruise) exports the guidance command (`meta["cmd_lat_accel"]`,
alongside ACTUATOR_LAG_S). The other generators prescribe acceleration directly with no actuator
lag, so commanded and achieved acceleration coincide and the signal used is the achieved lateral
acceleration derived from V. The table reports the source of each row, since a lead measured
against a command and a lead measured against an achieved state are different quantities.

Trim is estimated up to a change-point boundary. A fixed leading window (e.g. 10% of duration) is
unstable on this set: the HGV peak changes by 1.80x with window choice (the HGV weaves from t=0, so
a leading window contains maneuver and the trim absorbs signal), and the MaRV onset count runs
1->2->0->1 across window fractions. `--frac-sweep` reports the change-point result over a range of
boundary fractions.

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
    """Generate one trajectory of class `name` along the same path as the dataset generator.

      Corridor -- launch/target come from corridors_for(name), the class-eligible corridor list,
        picked uniformly. Range and geography drive the bisection solve for launch speed in
        ballistic and marv.
      Endpoints -- corridor.sample_endpoints(rng) rejection-samples against a land mask.
      Parameters -- experiments.generate_dataset._sample_params. It sets two values that differ
        from the generator defaults: supersonic_cruise boost_from_ground=True, and hgv aero="cavh"
        (wind-tunnel CAV-H polynomials, with lift that saturates at low dynamic pressure).
      MIRV -- generated with sim.mirv.mirv_bus. GENERATORS['mirv'] is a single-object adapter that
        returns only rv0 and omits the bus, the other two RVs and the three decoys.
      Physical-envelope check and retry -- audit_dataset.audit_trajectory, up to max_attempts draws, skipped
        when gate=False. Trajectories the dataset rejects are redrawn: for example the HGV 42 g
        heading-correction spike fails lateral_g, and MaRV pull-ups that flatten into a loft fail
        altitude.
      evasive -- left unset so the generator default True applies, as in the dataset
        (generate_scenario passes no evasive kwarg).

    Returns dict(name, t, P, V, dt, meta, attempts, corridor); for MIRV also objects, birth_step and
    death_step. Raises RuntimeError when the retry budget is exhausted, matching the dataset, which
    drops such a scenario.
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
                    last = "envelope checks"
                    continue
            # single-track view for the onset analysis: the primary RV, with the full object list
            # returned in `objects`
            rv = [o for o in objs if o.get("type") == "rv"] or objs
            o = rv[0]
            # Slice to the object's alive window. Every MIRV object carries full-length P/V arrays
            # in which samples before birth_step are the bus track (_assemble() co-locates each
            # object with the bus until dispense). The unsliced array includes the bus boost, with
            # peak |A| at step 0 of 5509 m/s^2 (561.8 g) for every object. Within the alive window
            # the dispense impulse is 14.6-18.1 g across the three RVs, consistent with the 17 g
            # listed in profiles.py. audit_trajectory uses the same slice(birth_step, death_step+1).
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
                last = "envelope checks"
                continue
        return dict(name=name, t=np.arange(len(P)) * dt, P=P, V=V, dt=float(dt),
                    meta=meta or {}, attempts=attempt + 1, corridor=corridor.name)
    raise RuntimeError("%s: %d attempts exhausted (%s) -- the dataset would drop this scenario"
                       % (name, max_attempts, last))


def lateral_signal(rec):
    """Per-sample lateral acceleration and its source. Returns (signal, source).

    'command' -- meta['cmd_lat_accel'], the guidance command. Only `maneuvering` exports this,
                 and it is the quantity a command-derived cue is built from.
    'achieved' -- the component of dV/dt perpendicular to velocity. This is what a sensor can in
                 principle observe, and for every generator except maneuvering it equals the
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
    """Split a signal into trim and maneuver at a change-point boundary.

    Seeds the trim from a short prefix (2% of samples, at least 3), finds the first sample whose
    deviation exceeds `frac` of the trajectory's own peak deviation, and takes the trim as the median
    before that sample. A fixed-length leading window can create or remove an onset depending on
    where it falls relative to the maneuver. Returns (|sig - trim|, trim, cut index).
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
    """Altitude and airframe lateral-acceleration limit along the trajectory.

    Returns (alt, amax) per sample, with amax = sim.sixdof.a_max_at(alt, Mach) in m/s^2 and NaN
    outside 0-90 km. A class whose command exceeds a_max saturates the airframe, and a lead measured
    on a saturated maneuver is not meaningful. For example, driving PitchAirframe with the HGV
    command at its 40 km glide puts 91% of the flight above the local a_max, with achieved a_z held
    at ~1.03 g against an 8.0 g command.

    The table at the top of trajectory_generators/profiles.py lists peak lateral demand per class
    against a lift-limited vehicle: supersonic_cruise 13.2 g (14.3x the evader aero-cap surrogate,
    4.7x a sourced CAV-H), hgv 42.4 g (14.6x / 5.8x), marv 15.0 g (12.8x / 4.2x). This function
    recomputes the limit from the trajectory.
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

    # Onset criterion: a quiescent phase (more than 20% of samples below threshold) followed by a
    # first departure after t=0. Later crossings are allowed, since the terminal weave of
    # supersonic_cruise is periodic and crosses the threshold repeatedly. The paper scores the
    # first crossing.
    i_on = int(np.argmax(above)) if above.any() else -1
    # Exo-atmospheric guard: above 80 km there is no dynamic pressure to turn against, so a
    # threshold crossing there is not an aerodynamic maneuver. Such crossings (for example a MIRV
    # dispense impulse, the modelled 140 m/s separation read through np.gradient) are flagged EXO.
    exo = bool(i_on > 0 and alt[i_on] > 80000.0)
    has_onset = bool(i_on > 0 and quiet > 0.2 and not exo)
    t_onset = float(t[i_on]) if i_on > 0 else None
    # altitude at onset (at launch when there is no crossing)
    alt_on = float(alt[i_on] / 1000) if i_on > 0 else float(alt[0] / 1000)

    # The a_max exceedance is computed only for rows whose signal is the guidance command; other
    # rows report n/a. Ballistic and MIRV are not aerodynamically driven (gravity and drag; MIRV's
    # 17 g is the dispense impulse), so a comparison against a fin-generated a_max does not apply.
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
                    help="sweep the change-point fraction and report the onset verdict at each value")
    args = ap.parse_args()

    names = list(GENERATORS.keys())

    if args.frac_sweep:
        print("Onset verdict across change-point fractions\n")
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
        print("\nEach cell is crossings/onset (Y or n). A constant row means the verdict does not depend on the fraction.")
        return

    print("Per-class lateral profile and aerodynamic feasibility")
    print("  classes enumerated from the registry trajectory_generators.profiles.GENERATORS")
    print("  (every registered class has a row).\n")
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

    print("\nColumn key")
    print("  signal   -- 'command' is the guidance command (only maneuvering exports it);")
    print("              'achieved' is the observable acceleration perpendicular to velocity.")
    print("  onset?   -- YES: quiescent phase followed by a first departure above the onset threshold;")
    print("              EXO: first crossing above 80 km; no: no qualifying onset. With 0 crossings")
    print("              the class never maneuvers and has no defined lead.")
    print("  > a_max  -- fraction of the flight commanding more lateral acceleration than the")
    print("              airframe can deliver at that altitude and Mach. A high value means the")
    print("              command saturates the airframe; n/a for rows without a guidance command.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
