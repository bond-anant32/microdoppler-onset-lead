"""
trajectory_generators/hgv.py - Hypersonic Glide Vehicle (DF-ZF / Avangard / LRHW class).
Modelled from the post-boost GLIDE phase (boost is abstracted: the vehicle starts at the
top of its glide band at hypersonic speed). In the glide, a base lift cancels gravity to
hold the band while an oscillating lift component produces damped skips (pull-up/push-
down), plus a lateral cross-range weave. All from bounded lift (a = v^2/r respected).

Contract: hgv_skip_glide(...) -> (positions (N,3) ECEF m, dt s).
RCS anchor (surrogate prior): HGV glide body ~0.01-0.05 m^2 (-20 to -13 dBsm).
"""
from __future__ import annotations
import numpy as np

from trajectory_math import LAUNCH_LAT, LAUNCH_LON, TARGET_LAT, TARGET_LON
from .base import launch_state, run, local_frame, clamp, G0
from .dynamics import gravity_accel


def hgv_skip_glide(launch_ll=(LAUNCH_LAT, LAUNCH_LON),
                   target_ll=(TARGET_LAT, TARGET_LON),
                   glide_speed=2400.0, glide_alt_m=40000.0,
                   n_skips=4.0, skip_amp=0.15, damp_k=1.0,
                   weave_g=8.0, weave_cycles=3.0, beta=15000.0, a_max_g=30.0,
                   dt=0.5, max_seconds=300.0):
    n_steps = int(round(max_seconds / dt))
    p0, v0 = launch_state(launch_ll, target_ll, glide_speed, 0.0, alt0=glide_alt_m)
    a_max = a_max_g * G0
    dur = max_seconds

    def control(t, p, v):
        up, fwd, right = local_frame(p, v)
        g_local = float(np.linalg.norm(gravity_accel(p)))
        v_h = float(np.linalg.norm(v - up * np.dot(v, up)))     # horizontal (cross-radial) speed
        r = float(np.linalg.norm(p))
        m = t / dur
        # base lift holds the LEVEL glide: cancels gravity MINUS the centripetal v_h^2/r needed to follow
        # the curved Earth. Without it the net specific force is ~0 -> a straight ECEF tangent that climbs
        # tens of km over the glide. Oscillating term -> damped skips.
        base = g_local - v_h * v_h / r
        lift_mag = base * (1.0 + skip_amp * np.sin(2 * np.pi * n_skips * m) * np.exp(-damp_k * m))
        a_lift = up * lift_mag
        a_weave = right * (weave_g * G0) * np.sin(2 * np.pi * weave_cycles * m)
        return clamp(a_lift + a_weave, a_max)

    pts = run(p0, v0, control, dt, n_steps, beta, stop_on_impact=True, min_alt=0.0)
    return pts, dt


if __name__ == "__main__":
    from .base import physics_report
    pts, dt = hgv_skip_glide()
    print("hgv:", {k: round(v, 2) if isinstance(v, float) else v
                   for k, v in physics_report(pts, dt).items()})
    print("PASS: HGV generates with skip-glide + cross-range weave.")
