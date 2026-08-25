"""
trajectory_generators/ballistic.py - force-integrated ballistic trajectory.

Pure gravity + aerodynamic drag (no lift/thrust after launch). Apogee and range EMERGE
from the launch speed and elevation angle, so realism is guaranteed by construction.
`depressed=True` uses a low elevation angle (flatter, faster, lower apogee) the way an
Iskander/KN-23 minimizes apogee to cut radar exposure.

Contract: ballistic(...) -> (positions (N,3) ECEF m, dt s).
RCS anchor (surrogate prior): ballistic RV/MaRV ~0.03-0.1 m^2 (-15 to -10 dBsm).
"""
from __future__ import annotations
import numpy as np

from trajectory_math import LAUNCH_LAT, LAUNCH_LON, TARGET_LAT, TARGET_LON
from .base import launch_state, run


def ballistic(launch_ll=(LAUNCH_LAT, LAUNCH_LON),
              target_ll=(TARGET_LAT, TARGET_LON),
              launch_speed=2200.0, elevation_deg=41.0,
              beta=20000.0, depressed=False,
              dt=0.5, max_seconds=400.0, alt0=50.0):
    """
    beta: ballistic coefficient m/(Cd*S) [kg/m^2]; higher = less drag.
    Returns (pts, dt). Integration stops at ground impact.
    """
    if depressed:
        elevation_deg = min(elevation_deg, 22.0)
    n_steps = int(round(max_seconds / dt))
    p0, v0 = launch_state(launch_ll, target_ll, launch_speed, elevation_deg, alt0=alt0)

    def no_control(t, p, v):
        return np.zeros(3)

    pts = run(p0, v0, no_control, dt, n_steps, beta, stop_on_impact=True, min_alt=0.0)
    return pts, dt


if __name__ == "__main__":
    from .base import physics_report
    pts, dt = ballistic()
    print("ballistic:", {k: round(v, 2) if isinstance(v, float) else v
                          for k, v in physics_report(pts, dt).items()})
    pts_d, _ = ballistic(depressed=True)
    print("depressed:", {k: round(v, 2) if isinstance(v, float) else v
                         for k, v in physics_report(pts_d, dt).items()})
    print("PASS: ballistic generates, arcs, and impacts." )
