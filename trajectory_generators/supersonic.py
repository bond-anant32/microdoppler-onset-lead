"""
trajectory_generators/supersonic.py - supersonic sea-skimming cruise missile (BrahMos /
P-800 / YJ-18 / Tsirkon class). Hi-lo profile: level cruise at ~14 km, Mach ~2.8 (lift
cancels gravity, thrust cancels drag), then a terminal DIVE to a sea-skim altitude with
an optional terminal weave. Lift is reduced (eased) in the dive so gravity pitches it
down; bounded throughout.

Contract: supersonic_cruise(...) -> (positions (N,3) ECEF m, dt s).
RCS anchor (surrogate prior): supersonic cruise ~0.5 m^2 (-3 dBsm; ramjet intake is a
strong scatterer broadside).
"""
from __future__ import annotations
import numpy as np

from trajectory_math import LAUNCH_LAT, LAUNCH_LON, TARGET_LAT, TARGET_LON
from .base import launch_state, run, local_frame, ease, clamp, G0
from .dynamics import gravity_accel, drag_accel


def supersonic_cruise(launch_ll=(LAUNCH_LAT, LAUNCH_LON),
                      target_ll=(TARGET_LAT, TARGET_LON),
                      cruise_speed=950.0, cruise_alt_m=14000.0,
                      terminal_start_frac=0.65, terminal_weave_g=6.0, weave_cycles=2.0,
                      dive_g=6.0, beta=8000.0, a_max_g=30.0, dt=0.5, max_seconds=220.0):
    n_steps = int(round(max_seconds / dt))
    p0, v0 = launch_state(launch_ll, target_ll, cruise_speed, 0.0, alt0=cruise_alt_m)
    a_max = a_max_g * G0
    dur = max_seconds
    t_dive = terminal_start_frac * dur

    def control(t, p, v):
        up, fwd, right = local_frame(p, v)
        g = gravity_accel(p)
        d = drag_accel(p, v, beta)
        v_h = np.linalg.norm(v - up * np.dot(v, up))      # horizontal (cross-radial) speed
        # centripetal term to hold the CURVED level-flight path; without it -g-d nets to ~0 and the
        # vehicle flies a straight ECEF tangent, climbing ~km over the cruise.
        a_cp = -up * (v_h ** 2 / np.linalg.norm(p))
        if t < t_dive:
            a = -g - d + a_cp                             # level, constant-speed cruise
        else:
            e = ease((t - t_dive) / (dur - t_dive))
            a = (-g * (1.0 - e)                           # bleed off lift -> gravity dives it
                 - d                                       # keep cancelling drag
                 + a_cp * (1.0 - e)                        # hold the level path until the dive takes over
                 - up * (dive_g * G0) * e                  # active downward push (sea-skim)
                 + right * (terminal_weave_g * G0) * e * np.sin(2 * np.pi * weave_cycles * e))
        return clamp(a, a_max)

    pts = run(p0, v0, control, dt, n_steps, beta, stop_on_impact=True, min_alt=0.0)
    return pts, dt


if __name__ == "__main__":
    from .base import physics_report
    pts, dt = supersonic_cruise()
    print("supersonic:", {k: round(v, 2) if isinstance(v, float) else v
                          for k, v in physics_report(pts, dt).items()})
    print("PASS: supersonic hi-lo cruise + terminal dive generated.")
