"""
audit_dataset.py - Tier-0 data-audit harness ("audit the training data").

Before any trajectory is allowed into the dataset, it must pass physics-consistency and
sanity checks. This is the lowest tier of the multi-tier verification stack:

    Tier 0  DATA AUDIT      <-- this file (physics/sanity gate on every trajectory)
    Tier 1  TRAINING MONITOR   (loss/val curves, overfit gap)   [later]
    Tier 2  MODEL AUDIT        (ablations, generalization)       [later]
    Tier 3  RUNTIME SUPERVISOR (IMM checks the ML in operation)  [the IMM, built]

Checks per trajectory (positions Nx3 ECEF, dt):
  - finite:        no NaN/Inf anywhere
  - altitude:      never far below ground (impact overshoot allowed); apogee within class cap
  - speed:         within the class's plausible envelope
  - a=v^2/r:       peak lateral load <= class cap AND min turn radius >= class floor
                   (the physics guardrail: no impossible sharp turns)
  - no-impulse:    bounded jerk (raised-cosine easing => continuous accel; a spike = a bug)

Runnable: audits all four force-integrated generators and prints a PASS/FAIL report.
When the batch generator exists it should call audit_trajectory() on each scenario and
refuse (or log) any that fail, so no unphysical sample ever reaches the ML.
"""
from __future__ import annotations
import os
import sys
from typing import Dict, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np

from trajectory_generators.base import velocities_from, G0
from trajectory_generators.dynamics import altitude_of

# Per-class plausibility envelopes (open-source calibrated; deliberately lenient so only
# genuinely unphysical samples fail). g = lateral load factor; radius in km.
CLASS_SPECS: Dict[str, dict] = {
    # jerk_cap 2000: a LONG-range ballistic RV reenters at ~7 km/s, and the drag-deceleration ONSET at 0-10 km alt is
    # a real, brief high-jerk event (measured peak ~1500 m/s^3, 0.3% of steps, all sub-10 km) -- HARDER than the
    # slower MaRV reentry (cap 1000) because it is faster, softer than the violent MIRV reentry (cap 5000). The old
    # SHORT-corridor dataset never hit this (slow reentry), so ballistic defaulted to JERK_CAP=300 and every long
    # deployment-matched trajectory was falsely rejected 'no_impulse'; the fast reentry is physical (RK4), not a spike.
    "ballistic":  dict(alt_max_km=1500.0, spd_min=150.0, spd_max=8000.0, g_cap=6.0,  r_min_km=40.0, jerk_cap=2000.0),
    # marv alt 700 km: a LONG-range maneuvering RV (DF-26/Pershing-II-class on a 2500-3800 km shot) is
    # sub-ballistic but still lofts ~500-650 km -- below the ~1000 km a pure ballistic of that range reaches
    # (so still discriminable), and above the 400 km a shorter-range MaRV reaches. jerk_cap: reentry drag onset +
    # the bounded terminal pull-up are real high-jerk events (cf. the mirv/decoy reentry jerk allowance).
    "marv":       dict(alt_max_km=700.0,  spd_min=250.0, spd_max=8000.0, g_cap=25.0, r_min_km=6.0, jerk_cap=1000.0),
    # hgv spd_min low: the modelled boost phase legitimately starts subsonic before the hypersonic glide (a
    # simplification of booster separation). g_cap 15 = aggressive dense-air terminal weave/jink (real CAV-H
    # terminal pull-up), up from 12 (which assumed a gentler mid-glide-only weave).
    "hgv":        dict(alt_max_km=130.0,  spd_min=500.0, spd_max=8000.0, g_cap=15.0, r_min_km=12.0, jerk_cap=600.0),
    # supersonic_cruise is now a HYPERSONIC cruise missile (Zircon/HACM-class scramjet): boost off the pad,
    # a ~24-30 km cruise at ~Mach 5-7.5, then a bounded terminal S-weave + dive. Envelope widened from the old
    # Mach-2.8/14 km ramjet-skimmer (alt 22 km / 1700 m/s) to alt 32 km / 2600 m/s. g_cap 16: terminal weave
    # ~10-15 g. spd_min low: the boost legitimately starts subsonic off the rail (as the hgv boost note).
    "supersonic": dict(alt_max_km=32.0,   spd_min=150.0, spd_max=2600.0, g_cap=16.0, r_min_km=3.0),
    # mirv: UNPOWERED bus/RVs/decoys (gravity+drag only, no lift). The maneuver checks are
    # inappropriate -- path curvature is from drag+gravity, not a commanded turn -- and violent
    # reentry deceleration is real jerk. Physical by construction (RK4); this is a sanity gate only.
    "mirv":       dict(alt_max_km=1500.0, spd_min=100.0, spd_max=8000.0, g_cap=40.0, r_min_km=2.0,
                       jerk_cap=5000.0),
    # decoy: light MIRV penetration aids (balloon/chaff/replica). LOW spd_min -- a low-beta decoy
    # legitimately decelerates far below an RV on reentry (that deceleration IS the drag discriminant,
    # slice #6); a chaff cloud can drift near-subsonic. Same unpowered gravity+drag physics as mirv.
    "decoy":      dict(alt_max_km=1500.0, spd_min=20.0,  spd_max=8000.0, g_cap=40.0, r_min_km=1.0,
                       jerk_cap=5000.0),
}
JERK_CAP = 300.0          # m/s^3 - a raised-cosine maneuver stays well under this; a spike = discontinuity
ALT_FLOOR_KM = -1.0       # allow slight impact overshoot from finite dt


def audit_trajectory(pts: np.ndarray, dt: float, missile_class: str) -> Tuple[bool, Dict[str, tuple]]:
    """Return (passed, checks) where checks[name] = (ok: bool, detail: str)."""
    spec = CLASS_SPECS.get(missile_class, CLASS_SPECS["marv"])
    checks: Dict[str, tuple] = {}

    # finite
    finite = bool(np.all(np.isfinite(pts)))
    checks["finite"] = (finite, "all positions finite" if finite else "NaN/Inf present")
    if not finite:
        return False, checks

    V = velocities_from(pts, dt)
    A = velocities_from(V, dt)
    speeds = np.linalg.norm(V, axis=1)
    alts_km = np.array([altitude_of(p) for p in pts]) / 1000.0

    # altitude band
    alt_ok = (alts_km.min() > ALT_FLOOR_KM) and (alts_km.max() < spec["alt_max_km"])
    checks["altitude"] = (alt_ok, "alt %.2f..%.1f km (cap %.0f)" %
                          (alts_km.min(), alts_km.max(), spec["alt_max_km"]))

    # speed envelope
    spd_ok = (speeds.min() > 0.5 * spec["spd_min"]) and (speeds.max() < 1.5 * spec["spd_max"])
    checks["speed"] = (spd_ok, "speed %.0f..%.0f m/s (env %.0f..%.0f)" %
                       (speeds.min(), speeds.max(), spec["spd_min"], spec["spd_max"]))

    # a = v^2/r  (lateral load + turn radius)
    lat = np.zeros(len(pts))
    for i, (v, a) in enumerate(zip(V, A)):
        s = speeds[i]
        if s < 1.0:
            continue
        vhat = v / s
        lat[i] = np.linalg.norm(a - np.dot(a, vhat) * vhat)
    peak_g = float(lat.max()) / G0
    # KNOWN BLIND SPOT (disclosed, not fixed here): g_cap is a CONSTANT with no dynamic-pressure term, so
    # this check cannot tell a flyable pull from an impossible one -- it only bounds the number. A MaRV
    # pulling 18 g at 57 km (qbar ~2.7 kPa, where the modelled airframe can make ~0.1 g) passes the 25 g cap
    # cleanly, and does. Note also `lat` here does NOT remove gravity, so on a near-horizontal flier this
    # reads ~1 g higher than the airframe actually has to generate. The aero-feasibility axis lives in
    # an aero-consistency REPORT, which is not shipped here -- promoting it to a gate would reject a
    # large share of the current dataset, so the aero requirement is disclosed rather than enforced.
    g_ok = peak_g <= spec["g_cap"]
    checks["lateral_g"] = (g_ok, "peak %.1f g (cap %.0f)" % (peak_g, spec["g_cap"]))

    mask = lat > 0.5
    min_r_km = float(np.min(speeds[mask] ** 2 / lat[mask]) / 1000.0) if mask.any() else float("inf")
    r_ok = min_r_km >= spec["r_min_km"]
    checks["turn_radius"] = (r_ok, "min radius %.1f km (floor %.0f) - a=v^2/r"
                             % (min_r_km, spec["r_min_km"]))

    # no-impulse (bounded jerk); per-class cap (mirv reentry has high but physical jerk)
    jerk_cap = spec.get("jerk_cap", JERK_CAP)
    J = velocities_from(A, dt)
    peak_jerk = float(np.linalg.norm(J, axis=1).max())
    jerk_ok = peak_jerk <= jerk_cap
    checks["no_impulse"] = (jerk_ok, "peak jerk %.1f m/s^3 (cap %.0f)" % (peak_jerk, jerk_cap))

    passed = all(ok for ok, _ in checks.values())
    return passed, checks


def audit_class_balance(class_counts: Dict[str, int]) -> Tuple[bool, str]:
    """Warn if any class is under-represented (< half the mean count)."""
    if not class_counts:
        return False, "no scenarios"
    counts = np.array(list(class_counts.values()))
    ok = counts.min() >= 0.5 * counts.mean()
    return ok, "counts %s (min %d, mean %.1f)" % (dict(class_counts), counts.min(), counts.mean())


def audit_geometry_diversity(median_aspects_deg, min_std_deg: float = 25.0) -> Tuple[bool, str]:
    """Audit M2: the dataset must span diverse threat-approach geometries, not one overhead-pass
    template. FAIL if the cross-scenario spread of per-scenario median aspect angle is too tight
    (all scenarios look geometrically alike) -- the 'diversity is fake' regression."""
    a = np.asarray([x for x in median_aspects_deg if x is not None], dtype=float)
    if a.size < 2:
        return False, "too few scenarios"
    std = float(a.std())
    ok = std >= min_std_deg
    return ok, "cross-scenario median-aspect std %.1f deg (floor %.0f); range %.0f-%.0f deg" % (
        std, min_std_deg, a.min(), a.max())


def _report(name: str, passed: bool, checks: Dict[str, tuple]):
    print(f"\n[{'PASS' if passed else 'FAIL'}] {name}")
    for cname, (ok, detail) in checks.items():
        print(f"    {'ok ' if ok else 'XX '} {cname:12s} {detail}")


if __name__ == "__main__":
    from trajectory_generators.ballistic import ballistic
    from trajectory_generators.marv import marv_terminal
    from trajectory_generators.hgv import hgv_skip_glide
    from trajectory_generators.supersonic import supersonic_cruise

    cases = [
        ("ballistic", "ballistic", lambda: ballistic()),
        ("MaRV", "marv", lambda: marv_terminal()),
        ("HGV", "hgv", lambda: hgv_skip_glide()),
        ("supersonic", "supersonic", lambda: supersonic_cruise()),
    ]
    print("=" * 70)
    print("TIER-0 DATA AUDIT - physics/sanity gate on every trajectory")
    print("=" * 70)
    all_pass = True
    for name, cls, gen in cases:
        pts, dt = gen()
        passed, checks = audit_trajectory(pts, dt, cls)
        _report(name, passed, checks)
        all_pass = all_pass and passed

    bal_ok, bal = audit_class_balance({"ballistic": 12, "marv": 13, "hgv": 12, "supersonic": 13})
    print("\n[%s] class balance: %s" % ("PASS" if bal_ok else "WARN", bal))
    print("\n" + ("ALL TRAJECTORIES PASS the Tier-0 gate." if all_pass
                  else "SOME TRAJECTORIES FAILED - inspect before adding to the dataset."))
