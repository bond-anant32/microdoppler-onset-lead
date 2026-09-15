# trajectory_generators/maneuvering.py
# Force-integrated supersonic-cruise profile: a near-level cruise followed by a
# bounded lateral S-weave in the terminal phase. A modelled autopilot holds the cruise
# level (lift cancels gravity, thrust cancels drag), so the net force during cruise is
# ~zero and velocity is constant, which is the regime a constant-velocity EKF assumes.
# The terminal weave adds a bounded lateral acceleration that the CV model does not predict.
#
# Maneuver magnitudes are bounded and satisfy a = v^2/r by construction: the model
# commands a lateral acceleration and the turn radius r = v^2/a_lat follows. The weave
# is high-g with a large turn radius, as reported for HGV/MaRV maneuvers.
import os

import numpy as np

from trajectory_math import ll_to_ecef, LAUNCH_LAT, LAUNCH_LON, TARGET_LAT, TARGET_LON
from .dynamics import (
    gravity_accel, drag_accel, integrate, altitude_of, MU_EARTH, induced_drag_accel,
)

# Target autopilot/airframe response lag tau_T (s). The vehicle's lateral acceleration follows
# the guidance command after this delay, so the kinematic maneuver lags the control input. The
# micro-Doppler maneuver-anticipation cue reads the command, which precedes the velocity bend by
# tau_T.
#
# tau_T is the closed-loop autopilot/airframe response constant; fin servo time constants are
# tens of ms. Values of 0.2-2 s are standard for a target autopilot in the pursuit-evasion and
# differential-game guidance literature, and the default 1.2 s lies in that range. The value is a
# modelling assumption.
# Override for sensitivity studies:  ACTUATOR_LAG_S=0.6 python experiments/onset_snr_sweep.py ...
# (the env var is read at import, so profiles.py's `from .maneuvering import ACTUATOR_LAG_S` sees it).
ACTUATOR_LAG_S = float(os.environ.get("ACTUATOR_LAG_S", "1.2"))


def _initial_heading(launch_ecef, target_ecef):
    """Unit horizontal velocity direction at launch, tangent to the surface,
    pointing toward the target along the great circle."""
    up = launch_ecef / np.linalg.norm(launch_ecef)
    to_target = target_ecef - launch_ecef
    tangent = to_target - up * np.dot(to_target, up)
    return tangent / np.linalg.norm(tangent)


def cruise_with_terminal_weave(
    launch_ll=(LAUNCH_LAT, LAUNCH_LON),
    target_ll=(TARGET_LAT, TARGET_LON),
    cruise_alt_m=26_000.0,         # hypersonic cruise band (Zircon/HACM class)
    cruise_speed_mps=2300.0,       # ~Mach 7
    beta=6000.0,                   # ballistic coefficient kg/m^2 (slender hypersonic body)
    duration_s=200.0,
    dt=0.2,
    boost_time=45.0,               # rocket boost from the ground up to cruise alt+speed (then scramjet cruise)
    weave_start_frac=0.6,
    weave_lat_accel=130.0,         # m/s^2 (~13 g) lateral, below the 16 g lateral-load cap; see the
    #                                docstring for aerodynamic feasibility at cruise altitude.
    weave_cycles=4.0,
    dive_start_frac=0.80,          # past this fraction the missile noses over and dives onto the target
    dive_tau=3.0,                  # velocity-steering time constant in the dive (s)
    dive_gmax=12.0,                # cap on dive-steering accel (g), below the 16 g lateral-load cap
    boost_from_ground=False,       # False: level cruise starting at cruise_alt_m with a time-based weave, no boost
    #                                or dive. True: boost from the ground up to the hypersonic cruise band, weave,
    #                                then dive onto the target (the setting used by the dataset generator).
):
    """
    Returns (positions (N,3) ECEF m, velocities (N,3) m/s, dt, meta) where meta
    carries the weave window so downstream code can label flight_state / shade plots.

    Aerodynamic feasibility. weave_lat_accel is commanded as a free lateral acceleration. It is bounded
    by a constant and does not scale with dynamic pressure, so at cruise altitude the weave amplitude
    sets the required lift loading (the HGV and MaRV generators exceed the same cap through their heading
    term and arming altitude respectively):

        default   13.2 g @ 25.2 km M5.5 (qbar 52 kPa) -> needs m/(S*CL) ~ 402 kg/m^2 = 14.3x evader surrogate, 4.7x CAV-H
        dataset   15.6 g @ 27.3 km M5.8 (qbar 41 kPa) -> needs m/(S*CL) ~ 270 kg/m^2 = 21.3x evader surrogate, 7.1x CAV-H

    The ratios compare against the CAV-H-like evader aero-cap surrogate and the sourced CAV-H fit. The
    trajectory passes the physical-envelope checks (16 g lateral cap, no dynamic-pressure term). 13 g becomes
    feasible below ~8 km with the evader surrogate and below ~15 km with CAV-H aerodynamics, while the
    generator weaves at 25-27 km. A Zircon/HACM-class cruise missile is a winged body with a lower lift
    loading than a CAV-H waverider, and 270-400 kg/m^2 is plausible for a winged missile. No sourced
    lift model for this class is included.
    """
    launch_surf = ll_to_ecef(*launch_ll, 0.0)
    target_surf = ll_to_ecef(*target_ll, 0.0)
    up0 = launch_surf / np.linalg.norm(launch_surf)
    vhat0 = _initial_heading(launch_surf, target_surf)
    R_e = np.linalg.norm(launch_surf)
    ang = np.arccos(np.clip(np.dot(up0, target_surf / np.linalg.norm(target_surf)), -1.0, 1.0))
    ground_dist = ang * R_e
    cruise_time = ground_dist / max(cruise_speed_mps, 1.0)
    boost_accel = cruise_speed_mps / max(boost_time, 1.0) + 12.0  # thrust to reach cruise speed by burnout
    dive_range = 55_000.0                                   # nose over when within 55 km ground-track distance of the target
    heading_gain = 0.12                                     # gentle cruise heading correction toward target (level-cruise mode)

    if boost_from_ground:
        # Boost from the pad up to cruise, cruise the great-circle distance, then dive.
        p0 = launch_surf + up0 * 50.0
        v0 = up0 * 160.0                                    # leaves the rail moving up (avoids the low-speed transient)
        max_time = max(boost_time + cruise_time + 60.0, duration_s)
        weave_dur = float(np.clip((1.0 - weave_start_frac) * cruise_time, 20.0, 60.0))
        weave_start_t = max(boost_time + cruise_time - weave_dur, boost_time + 5.0)
        arrive_t = boost_time + cruise_time
    else:
        # Level-cruise mode: start in the cruise band flying level and weave over the flight, with no boost or
        # dive. The cruise has ~zero net force.
        p0 = launch_surf + up0 * cruise_alt_m
        v0 = cruise_speed_mps * vhat0
        max_time = duration_s
        weave_dur = float(np.clip((1.0 - weave_start_frac) * duration_s, 20.0, 60.0))
        weave_start_t = weave_start_frac * duration_s
        arrive_t = duration_s + 1.0                         # weave allowed through the end (no terminal dive)
    n_steps = int(round(max_time / dt))
    dive_gcap = dive_gmax * 9.80665
    heading_gain = 0.12                                     # gentle cruise heading correction toward target

    def a_lat_command(t):
        """Commanded lateral acceleration (m/s^2): the autopilot's control input."""
        if t < weave_start_t or t > arrive_t:
            return 0.0
        m = (t - weave_start_t) / weave_dur
        return weave_lat_accel * np.sin(2 * np.pi * weave_cycles * m)

    # ---- command -> response model -------------------------------------------------
    # LAG_MODE=delay : a(t) = a_cmd(t - tau)          pure transport delay (default)
    # LAG_MODE=lag   : da/dt = (a_cmd(t) - a(t))/tau  first-order lag (the differential-game guidance
    #                  convention, used e.g. by Mudrik & Oshman)
    # At the same tau the first-order lag begins responding immediately and crosses a small detection
    # threshold ~1 s earlier than the pure delay, so the pure delay gives the largest anticipation
    # window for a given tau. The response model therefore affects any measured anticipation window.
    _LAG_MODE = os.environ.get("LAG_MODE", "delay").lower()
    _lag_t = _lag_a = None
    if _LAG_MODE == "lag" and ACTUATOR_LAG_S > 1e-9:
        # Precompute the first-order response on a fine grid, then interpolate (accel_fn is called
        # at arbitrary RK4 sub-step times, so a stateful filter inside it would be order-dependent).
        _h = min(0.01, float(dt) / 4.0)
        _n = int(np.ceil((arrive_t + 5.0) / _h)) + 2
        _lag_t = np.arange(_n) * _h
        _lag_a = np.zeros(_n)
        _k = _h / ACTUATOR_LAG_S
        for _i in range(1, _n):                       # explicit Euler on da/dt=(cmd-a)/tau
            _lag_a[_i] = _lag_a[_i - 1] + _k * (a_lat_command(_lag_t[_i - 1]) - _lag_a[_i - 1])

    def a_lat_applied(t):
        """Lateral acceleration (m/s^2) produced by the airframe at time t."""
        if _LAG_MODE == "lag" and _lag_a is not None:
            return float(np.interp(t, _lag_t, _lag_a))
        return a_lat_command(t - ACTUATOR_LAG_S)

    def accel_fn(t, p, v):
        a_gravity = gravity_accel(p)
        a_drag = drag_accel(p, v, beta)
        r = np.linalg.norm(p)
        up = p / r
        speed = np.linalg.norm(v)
        vhat = v / speed if speed > 1e-6 else up
        v_h = np.linalg.norm(v - up * np.dot(v, up))        # horizontal (cross-radial) speed
        to_tgt = target_surf - p
        dist_tgt = np.linalg.norm(to_tgt)
        horiz_dist = np.linalg.norm(to_tgt - up * np.dot(to_tgt, up))   # ground-track distance (alt-independent)

        # Terminal dive within dive_range (horizontal) of the target: nose over onto it and impact. The trigger
        # uses ground-track distance because the 3-D range is at least the cruise altitude even directly
        # overhead, so a small 3-D threshold would fire only on an exact overflight. The velocity is steered at
        # the target with bounded acceleration so the pull-over stays within the turn-radius floor;
        # gravity brings the vehicle down, and the sustaining motor cancels drag so speed holds into the dive.
        if boost_from_ground and horiz_dist < dive_range:
            desired = to_tgt / dist_tgt if dist_tgt > 1e-6 else vhat
            if float(np.dot(desired, vhat)) < 0.2:              # overflew the target -> dive to nadir, don't loop back
                desired = -up
            a_turn = (desired - vhat) * (speed / max(dive_tau, 1e-3))
            a_turn = a_turn - np.dot(a_turn, vhat) * vhat        # pure turn (perpendicular) -> preserves airspeed
            nrm = np.linalg.norm(a_turn)
            if nrm > dive_gcap:
                a_turn = a_turn * (dive_gcap / nrm)
            return a_gravity + a_turn                           # motor cancels drag; gravity noses it down

        alt = altitude_of(p)
        vz = float(np.dot(v, up))
        tgt_h = to_tgt - up * np.dot(to_tgt, up)
        nth = np.linalg.norm(tgt_h)
        head = (tgt_h / nth) if nth > 1e-6 else vhat

        if boost_from_ground:
            # Gravity-turn boost: while climbing to the cruise band, thrust along an altitude-scheduled pitch
            # that rotates from vertical to the target heading, so thrust stays roughly along velocity and the
            # lateral load stays low.
            if alt < cruise_alt_m and speed < cruise_speed_mps:
                frac = float(np.clip(alt / (0.7 * cruise_alt_m), 0.0, 1.0))   # pitch to horizontal by 70% cruise alt
                d = (1.0 - frac) * up + frac * head
                dhat = d / np.linalg.norm(d)
                return a_gravity + a_drag + dhat * boost_accel
            # Cruise hold: level at cruise_alt/cruise_speed (P/D altitude + bounded forward thrust), cancelling
            # gravity, drag and the centripetal term so the held cruise has ~zero net force.
            a_fwd = head * float(np.clip(0.05 * (cruise_speed_mps - speed), -60.0, 90.0))
            a_climb = up * (0.004 * (cruise_alt_m - alt) - 0.35 * vz)
            a_ctrl = -a_gravity - a_drag - up * (v_h ** 2 / r) + a_fwd + a_climb
        else:
            # Level cruise: lift cancels gravity and thrust cancels drag, less the centripetal term the vehicle
            # needs to hold the curved level path, giving ~zero net force (constant velocity). A gentle heading
            # correction toward the target keeps it on the great circle.
            a_ctrl = -a_gravity - a_drag - up * (v_h ** 2 / r)
            if nth > 1e-6 and t < weave_start_t:
                vh_vec = v - up * np.dot(v, up)
                nvh = np.linalg.norm(vh_vec)
                vhat_h = vh_vec / nvh if nvh > 1e-6 else head
                a_head = (head - vhat_h) * (speed * heading_gain)
                a_head = a_head - np.dot(a_head, vhat) * vhat        # pure turn: corrects heading without bleeding speed
                a_ctrl = a_ctrl + a_head

        # Terminal lateral S-weave, applied through the actuator lag, so the command read by the
        # micro-Doppler anticipation cue leads the velocity bend.
        a_applied = a_lat_applied(t)
        if a_applied != 0.0:
            cross = np.cross(up, vhat)
            ncross = np.linalg.norm(cross)
            if ncross > 1e-9:
                a_ctrl = a_ctrl + (cross / ncross) * a_applied
                # Induced drag: the S-weave is a lift force and costs drag (~a_lat^2), so a hard weave
                # bleeds cruise speed.
                a_ctrl = a_ctrl + induced_drag_accel(v, abs(float(a_applied)))

        return a_gravity + a_drag + a_ctrl

    P, V = integrate(p0, v0, accel_fn, dt, n_steps)

    # Truncate the trajectory at ground impact.
    alts = np.array([altitude_of(P[i]) for i in range(len(P))])
    below = np.where(alts <= 0.0)[0]
    if len(below):
        n_steps = min(len(P), int(below[0]) + 1)
        P, V = P[:n_steps], V[:n_steps]
        
    dist_to_tgt = np.linalg.norm(P - target_surf, axis=1)
    dive_steps = np.where(dist_to_tgt < dive_range)[0]
    dive_start_step = int(dive_steps[0]) if len(dive_steps) > 0 else n_steps - 1

    # Record the command schedule (independent input) so the signature layer can drive the
    # anticipation cue from it; the kinematic response lags it by ACTUATOR_LAG_S.
    cmd_lat_accel = np.array([abs(a_lat_command(i * dt)) for i in range(n_steps)])

    meta = {
        "dt": dt,
        "n_steps": n_steps,
        "weave_start_frac": weave_start_frac,
        "weave_start_step": min(int(weave_start_t / dt), n_steps - 1),   # command onset (weave start, before the dive)
        "weave_lat_accel": weave_lat_accel,
        "cruise_speed_mps": cruise_speed_mps,
        "cruise_alt_m": cruise_alt_m,
        "turn_radius_m": cruise_speed_mps ** 2 / weave_lat_accel,
        "cmd_lat_accel": cmd_lat_accel,
        "actuator_lag_s": ACTUATOR_LAG_S,
        "dive_start_step": dive_start_step,
    }
    return P, V, dt, meta


def make_state_fn(P, V, dt):
    """Wrap integrated samples as a state_fn(t) -> (p, v), matching
    trajectory_math.state_fn, so the EKF/measurement pipeline can consume it."""
    n = len(P)
    duration = (n - 1) * dt

    def state_fn(t):
        t = max(0.0, min(float(t), duration))
        f = t / dt
        i = min(int(f), n - 2)
        frac = f - i
        p = P[i] + frac * (P[i + 1] - P[i])
        v = V[i] + frac * (V[i + 1] - V[i])
        return p, v

    return state_fn
