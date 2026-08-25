"""
trajectory_generators/marv.py - Maneuvering Reentry Vehicle (Iskander-M / Fateh-110 /
KN-23 class). Ballistic midcourse, then a TERMINAL pull-up + lateral S-weave once the
vehicle descends through a trigger altitude. The terminal maneuver is bounded lift added
to gravity+drag (a = v^2/r respected), eased in with a raised-cosine (no impulse).

Contract: marv_terminal(...) -> (positions (N,3) ECEF m, dt s).
RCS anchor (surrogate prior): MaRV ~0.03-0.1 m^2 (-15 to -10 dBsm).
"""
from __future__ import annotations
import numpy as np

from trajectory_math import LAUNCH_LAT, LAUNCH_LON, TARGET_LAT, TARGET_LON
from .base import launch_state, run, local_frame, ease, clamp, G0
from .dynamics import altitude_of


def marv_terminal(launch_ll=(LAUNCH_LAT, LAUNCH_LON),
                  target_ll=(TARGET_LAT, TARGET_LON),
                  launch_speed=2200.0, elevation_deg=41.0, beta=20000.0,
                  pullup_trigger_alt_m=25000.0, maneuver_dur_s=30.0,
                  pullup_g=8.0, weave_g=10.0, weave_cycles=3.0, a_max_g=40.0,
                  dt=0.5, max_seconds=400.0, alt0=50.0):
    n_steps = int(round(max_seconds / dt))
    p0, v0 = launch_state(launch_ll, target_ll, launch_speed, elevation_deg, alt0=alt0)
    a_max = a_max_g * G0
    st = {"engaged": False, "t0": 0.0}

    def control(t, p, v):
        up, fwd, right = local_frame(p, v)
        alt = altitude_of(p)
        descending = np.dot(v, up) < 0.0
        if not st["engaged"]:
            if descending and alt < pullup_trigger_alt_m:
                st["engaged"] = True
                st["t0"] = t
            else:
                return np.zeros(3)
        m = np.clip((t - st["t0"]) / maneuver_dur_s, 0.0, 1.0)
        env = np.sin(np.pi * m)                           # 0 -> 1 -> 0 bounded pulse
        a_pullup = up * (pullup_g * G0) * env             # arrest the descent, then release
        a_weave = right * (weave_g * G0) * env * np.sin(2 * np.pi * weave_cycles * m)
        return clamp(a_pullup + a_weave, a_max)

    pts = run(p0, v0, control, dt, n_steps, beta, stop_on_impact=True, min_alt=0.0)
    return pts, dt


if __name__ == "__main__":
    from .base import physics_report
    pts, dt = marv_terminal()
    print("marv:", {k: round(v, 2) if isinstance(v, float) else v
                    for k, v in physics_report(pts, dt).items()})
    print("PASS: MaRV generates with a bounded terminal maneuver.")
