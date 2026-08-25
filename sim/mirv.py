"""
sim/mirv.py - Force-integrated MULTI-OBJECT MIRV generator (bus -> N reentry vehicles + M decoys).

Rewritten from the old Euler/flat-gravity/spherical-altitude placeholder to the same force-
integrated fidelity as every other generator: RK4 of central WGS84 gravity + layered-atmosphere
drag (trajectory_generators.dynamics). A boosted ballistic BUS lofts to apogee, then DISPENSES:
  - N reentry vehicles (RVs): heavy (high ballistic coefficient beta), each given a small
    separation delta-v so they fly to spread aimpoints.
  - M decoys: light (LOW beta = high drag) -> they decelerate faster than the RVs once they hit
    the atmosphere (~below 80-100 km), so RVs pull ahead of decoys. That deceleration/ballistic-
    coefficient difference is the physical RV-vs-decoy DISCRIMINANT the tracker/ML must learn
    (decoys are indistinguishable in vacuum; drag separates them on reentry).

Returns a MULTI-OBJECT structure (not the single (P,V,dt,meta) contract) so the pipeline can track
several simultaneous objects:
    (objects, dt, meta)
  objects = [ {id, type in {bus,rv,decoy}, P (n_total,3), V (n_total,3), birth_step, death_step,
               spec_key, beta}, ... ]
Every object shares one length-n_total time grid: before its birth it sits on the bus track
(co-located); after its death (impact) it holds its last position. `birth_step..death_step` is its
alive window -- the measurement layer only emits detections while an object is alive.
"""
from __future__ import annotations
import os
import sys
import math
from typing import List, Dict, Tuple, Optional
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_math import ll_to_ecef
from trajectory_generators.dynamics import gravity_accel, drag_accel, integrate, altitude_of
from trajectory_generators.geo import ecef_to_geodetic


def _launch_frame(launch_ll, target_ll):
    launch_surf = ll_to_ecef(*launch_ll, 0.0)
    tgt_surf = ll_to_ecef(*target_ll, 0.0)
    up = launch_surf / np.linalg.norm(launch_surf)
    to_t = tgt_surf - launch_surf
    horiz = to_t - up * np.dot(to_t, up)
    horiz = horiz / np.linalg.norm(horiz)
    right = np.cross(horiz, up)                       # cross-range unit
    return launch_surf, up, horiz, right


def _integrate_to_impact(p0, v0, beta, dt, max_steps):
    """RK4 gravity+drag until ground impact; returns (P, V) truncated at first impact."""
    def accel(t, p, v):
        return gravity_accel(p) + drag_accel(p, v, beta)
    P, V = integrate(p0, v0, accel, dt, max_steps)
    alt = np.array([altitude_of(p) for p in P])
    hit = np.where(alt[2:] <= 0.0)[0]
    end = (hit[0] + 3) if len(hit) else max_steps
    return P[:end], V[:end]


def _great_circle_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    d = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, d)))   # was sqrt(sqrt(d)) -- extra inner sqrt (~2-4x too long)


def mirv_bus(launch_ll, target_ll, num_rv=3, num_decoy=3,
             launch_elev_deg=33.0, beta_bus=9000.0,
             beta_rv=9000.0, beta_decoy=900.0, sep_dv=140.0,
             dt=0.5, max_duration=2200.0, rng=None):
    """Force-integrated MIRV: boosted ballistic bus -> dispense at apogee -> N RVs + M decoys.
    Returns (objects, dt, meta). Range-matched to the launch->target great circle so the RVs
    reenter within the defended-side sensors' coverage. RVs spread to aimpoints; decoys (low beta)
    separate from RVs on reentry via drag (the discriminant)."""
    rng = rng or np.random.default_rng()
    launch_surf, up, horiz, right = _launch_frame(launch_ll, target_ll)
    elev = math.radians(launch_elev_deg)
    p0 = launch_surf + up * 50.0
    max_steps = int(max_duration / dt)
    rng_km = _great_circle_km(launch_ll, target_ll)

    # SHOOTING SOLVER: bisection on launch speed so the (drag-affected) bus impacts at the corridor
    # range -> the reentry vehicles land near the defended region and its radars can see them.
    def _fly(vmag):
        v0 = vmag * (math.cos(elev) * horiz + math.sin(elev) * up)
        Pb, Vb = _integrate_to_impact(p0, v0, beta_bus, dt, max_steps)
        la, lo, _ = ecef_to_geodetic(*Pb[-1])
        return _great_circle_km(launch_ll, (la, lo)), Pb, Vb

    # Widened bracket + reachability guard: the old [3000, 8000] m/s ceiling reached only ~818 km from an
    # impulsive sea-level launch (heavy t=0 drag), so ANY corridor beyond that pinned the speed at ~7995 and
    # the RVs landed ~800 km SHORT of the defended sensors -- silently breaking MIRV custody (
    # hunt). 14 km/s reaches ~2500 km, covering every corridor; the guard prevents silent saturation.
    lo_v, hi_v = 3000.0, 14000.0
    if _fly(hi_v)[0] < rng_km:
        launch_speed = hi_v                       # target genuinely beyond max reach -> best effort, not a silent pin
    else:
        for _ in range(12):                       # bisect launch speed to hit the corridor great-circle range
            mid = 0.5 * (lo_v + hi_v)
            if _fly(mid)[0] < rng_km:
                lo_v = mid
            else:
                hi_v = mid
        launch_speed = 0.5 * (lo_v + hi_v)
    _, Pb, Vb = _fly(launch_speed)
    alt_b = np.array([altitude_of(p) for p in Pb])
    dispense_step = int(np.argmax(alt_b))             # dispense at apogee (exo-atmospheric)
    dispense_step = min(max(dispense_step, 5), len(Pb) - 5)
    p_sep, v_sep = Pb[dispense_step], Vb[dispense_step]

    # --- spread pattern: RVs to different aimpoints, decoys near the RV cloud ---
    objects: List[Dict] = []

    def _spread_dv(k, n, mag):
        """A small separation impulse in the cross-range/along-range plane (fan-out)."""
        ang = (2.0 * math.pi * k / max(n, 1)) + float(rng.uniform(-0.3, 0.3))
        radial = math.cos(ang) * right + math.sin(ang) * horiz
        return mag * radial + up * float(rng.uniform(-0.2, 0.2)) * mag

    child_paths = []   # (type, beta, P, V)
    for k in range(num_rv):
        dv = _spread_dv(k, num_rv, sep_dv * float(rng.uniform(0.7, 1.3)))
        P, V = _integrate_to_impact(p_sep, v_sep + dv, beta_rv, dt, max_steps - dispense_step)
        child_paths.append(("rv", beta_rv, P, V))
    # decoy VARIETY (slice #6): dispense a MIX of decoy kinds spanning the discrimination-difficulty gradient.
    # beta (ballistic coeff, kg/m^2) is the reentry DRAG discriminant; micro-Doppler (sim/signatures.py) is the
    # spin/tumble discriminant. replica = HARD (beta near the RV's 9000 AND mimics RV spin); balloon/chaff = light
    # (low beta -> decelerate fast, fall behind the RVs) with distinct tumble rates.
    DECOY_KINDS = [("decoy_replica", 4800.0), ("decoy_balloon", 260.0), ("decoy_chaff", 45.0)]
    for k in range(num_decoy):
        kind, beta_k = DECOY_KINDS[k % len(DECOY_KINDS)]
        dv = _spread_dv(k, max(num_decoy, 1), sep_dv * float(rng.uniform(0.5, 1.1)))
        P, V = _integrate_to_impact(p_sep, v_sep + dv, beta_k, dt, max_steps - dispense_step)
        child_paths.append((kind, beta_k, P, V))

    # total timeline = dispense + longest child (bus debris also flies to impact)
    n_total = dispense_step + max([len(P) for _, _, P, _ in child_paths] + [len(Pb) - dispense_step])

    def _assemble(P_child, V_child, birth):
        """Full-length arrays: bus track before birth, own track after, hold last pos after death."""
        P = np.zeros((n_total, 3)); V = np.zeros((n_total, 3))
        # pre-birth: sit on the bus trajectory (co-located with the bus)
        pre = min(birth, len(Pb))
        P[:pre] = Pb[:pre]; V[:pre] = Vb[:pre]
        if pre < birth:                               # bus already impacted (shouldn't happen pre-apogee)
            P[pre:birth] = Pb[-1]; V[pre:birth] = 0.0
        # own trajectory
        m = len(P_child); end = min(birth + m, n_total)
        P[birth:end] = P_child[:end - birth]; V[birth:end] = V_child[:end - birth]
        death = birth + m - 1
        if end < n_total:                             # after impact: hold last position (dead)
            P[end:] = P_child[m - 1]; V[end:] = 0.0
        return P, V, death

    # bus object (flies from launch to its own impact)
    Pbus, Vbus, bus_death = _assemble(Pb[dispense_step:], Vb[dispense_step:], dispense_step)
    # actually the bus exists from step 0
    Pbus[:dispense_step] = Pb[:dispense_step]; Vbus[:dispense_step] = Vb[:dispense_step]
    objects.append(dict(id="bus", type="bus", P=Pbus, V=Vbus, birth_step=0,
                        death_step=min(dispense_step + (len(Pb) - dispense_step) - 1, n_total - 1),
                        spec_key="mirv", beta=beta_bus))

    ri = di = 0
    for typ, beta, P_child, V_child in child_paths:
        P, V, death = _assemble(P_child, V_child, dispense_step)
        if typ == "rv":
            oid = f"rv{ri}"; ri += 1
        else:
            oid = f"decoy{di}"; di += 1
        objects.append(dict(id=oid, type=typ, P=P, V=V, birth_step=dispense_step,
                            death_step=int(death),
                            spec_key=("mirv" if typ == "rv" else "decoy"),   # light decoys use the low-spd_min gate
                            beta=beta))

    meta = {
        "dt": dt, "n_steps": n_total, "missile_type": "mirv", "maneuver_class": "mirv",
        "dispense_step": dispense_step, "dispense_t": round(dispense_step * dt, 1),
        "dispense_alt_km": round(alt_b[dispense_step] / 1000.0, 1),
        "num_rv": num_rv, "num_decoy": num_decoy, "n_objects": len(objects),
        "peak_alt_m": float(alt_b.max()), "weave_start_step": None,
    }
    return objects, dt, meta
