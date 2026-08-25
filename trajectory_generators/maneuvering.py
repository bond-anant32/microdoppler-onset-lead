# trajectory_generators/maneuvering.py
# A force-integrated supersonic-cruise profile that flies a near-level cruise and
# then executes a bounded lateral S-weave in the terminal phase. The cruise is held
# level by a modelled autopilot: lift cancels gravity and thrust cancels drag, so
# the *net* force during cruise is ~zero (constant velocity) -- exactly the regime a
# CV-EKF assumes. The terminal weave injects a bounded lateral acceleration that the
# CV model cannot anticipate. This is the trajectory used to demonstrate EKF failure.
#
# All maneuver magnitudes are bounded and obey a = v^2/r by construction (we command
# a lateral acceleration; the turn radius r = v^2/a_lat emerges), so nothing here
# violates physics -- the weave is "high-G but large-radius", per HGV/MaRV reporting.
import os

import numpy as np

from trajectory_math import ll_to_ecef, LAUNCH_LAT, LAUNCH_LON, TARGET_LAT, TARGET_LON
from .dynamics import (
    gravity_accel, drag_accel, integrate, altitude_of, MU_EARTH, induced_drag_accel,
)

# TARGET AUTOPILOT / AIRFRAME RESPONSE LAG tau_T (s): the vehicle's lateral acceleration follows
# the guidance command after this delay, so the kinematic maneuver LAGS the control input. This is
# the physical basis of the micro-Doppler maneuver-anticipation cue — the cue reads the
# command, which precedes the velocity bend by tau_T.
#
# NAMING: this is the CLOSED-LOOP autopilot/airframe response constant tau_T, NOT a fin
# servo lag (servo time constants are tens of ms). tau_T ~ 0.2-2 s is the range standard in the
# pursuit-evasion / DGL guidance literature for a target autopilot; 1.2 s sits inside it. The value
# is CALIBRATED, NOT MEASURED — treat it as a modeling assumption and cite tau_T when reporting.
# Override for sensitivity studies:  ACTUATOR_LAG_S=0.6 python experiments/onset_snr_sweep.py ...
# (env var is read at import, so profiles.py's `from .maneuvering import ACTUATOR_LAG_S` sees it).
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
    cruise_alt_m=26_000.0,         # HYPERSONIC cruise band (Zircon/HACM class), up from the old Mach-2.8/14 km
    cruise_speed_mps=2300.0,       # ~Mach 7
    beta=6000.0,                   # ballistic coefficient kg/m^2 (slender hypersonic body)
    duration_s=200.0,
    dt=0.2,
    boost_time=45.0,               # rocket boost from the GROUND up to cruise alt+speed (then scramjet cruise)
    weave_start_frac=0.6,
    weave_lat_accel=130.0,         # m/s^2 (~13 g) lateral, bounded (headroom under the 16 g gate) -- but see
    #                                the AERO-CONSISTENCY DISCLOSURE in the docstring: 13 g is inside the g
    #                                gate and outside the modelled aero cap by ~14x at the cruise altitude.
    weave_cycles=4.0,
    dive_start_frac=0.80,          # past this fraction the missile noses over and dives onto the target
    dive_tau=3.0,                  # velocity-steering time constant in the dive (s) -- fast terminal weave/dive
    dive_gmax=12.0,                # cap on dive-steering accel (g) -- headroom under the 16 g gate
    boost_from_ground=False,       # False: legacy LEVEL cruise from cruise_alt + time-based weave (the tested,
    #                                backward-compatible default). True: boost from the GROUND (it RISES) up to
    #                                the hypersonic cruise band, then dive onto the target (the viz profile).
):
    """
    Returns (positions (N,3) ECEF m, velocities (N,3) m/s, dt, meta) where meta
    carries the weave window so downstream code can label flight_state / shade plots.

    AERO-CONSISTENCY DISCLOSURE (measured). weave_lat_accel is
    commanded as a free lateral acceleration -- bounded by a constant, never by dynamic pressure. Unlike the
    HGV (whose peak comes from an unbounded heading term) and the MaRV (whose gap comes from arming too
    high), THIS class's gap IS the documented weave amplitude, flown exactly as specified:

        default   13.2 g @ 25.2 km M5.5 (qbar 52 kPa) -> needs m/(S*CL) ~ 402 kg/m^2 = 14.3x evader.py, 4.7x CAV-H
        dataset   15.6 g @ 27.3 km M5.8 (qbar 41 kPa) -> needs m/(S*CL) ~ 270 kg/m^2 = 21.3x evader.py, 7.1x CAV-H

    It passes Tier-0 every time (cap 16 g, no qbar term), so this gap ships in full. Two honest readings,
    and the repo currently supports neither explicitly:
      (a) the ALTITUDE is wrong -- study/FLIGHT_PROFILE_SPEC.md line 45 puts the terminal S-weave "below
          ~20 km where aero authority (dynamic pressure) is high"; 13 g first becomes feasible at ~8 km
          (evader surrogate) / ~15 km (CAV-H). The generator weaves at 25-27 km instead.
      (b) the AIRFRAME is wrong -- guidance/evader.evader_aero_cap_g applies ONE CAV-H waverider surrogate
          to every class, but a Zircon/HACM-class cruise missile is a WINGED body with a much lower lift
          loading. 270-400 kg/m^2 is not an absurd number for a winged missile; it is absurd for a CAV-H.
          The repo has no sourced lift model for this class, so the requirement stays DISCLOSED and
          UNSOURCED rather than back-fitted.
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
    dive_range = 55_000.0                                   # nose over when within 55 km GROUND-TRACK of the target
    heading_gain = 0.12                                     # gentle cruise heading correction toward target (legacy mode)

    if boost_from_ground:
        # RISE from the pad: boost up to cruise, cruise the great-circle distance, then dive (viz profile).
        p0 = launch_surf + up0 * 50.0
        v0 = up0 * 160.0                                    # boosts off the rail moving UP (avoids the low-speed transient)
        max_time = max(boost_time + cruise_time + 60.0, duration_s)
        weave_dur = float(np.clip((1.0 - weave_start_frac) * cruise_time, 20.0, 60.0))
        weave_start_t = max(boost_time + cruise_time - weave_dur, boost_time + 5.0)
        arrive_t = boost_time + cruise_time
    else:
        # LEGACY (tested): start AT the cruise band flying level, weave over the flight, no boost/dive. This is
        # the ~zero-net-force level cruise the signature/property/level-flight invariants validate.
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
        """Commanded lateral acceleration (the control INPUT) — the pilot/autopilot's demand."""
        if t < weave_start_t or t > arrive_t:
            return 0.0
        m = (t - weave_start_t) / weave_dur
        return weave_lat_accel * np.sin(2 * np.pi * weave_cycles * m)

    # ---- command -> response model -------------------------------------------------
    # LAG_MODE=delay : a(t) = a_cmd(t - tau)          PURE TRANSPORT DELAY (legacy default)
    # LAG_MODE=lag   : da/dt = (a_cmd(t) - a(t))/tau  FIRST-ORDER LAG (the DGL/guidance convention,
    #                  used by Mudrik & Oshman and the whole differential-game literature)
    # These are NOT equivalent at the same tau: the first-order lag begins responding immediately
    # and crosses a small detection threshold ~1 s EARLIER than a pure delay of the same tau, so
    # the pure delay yields the LARGEST possible anticipation window for a given tau. Since the
    # paper's headline IS an anticipation window, this must be reported as an ablation, not assumed.
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
        """The lateral acceleration the AIRFRAME actually produces at time t."""
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

        # TERMINAL DIVE within dive_range (HORIZONTAL) of the target: nose over onto it and impact. Trigger on the
        # ground-track distance, not the 3-D range -- at a 28 km cruise the 3-D range is ~28 km even directly
        # overhead, so a small 3-D gate would fire only on a perfect overflight and otherwise the missile sails
        # past. Steer the velocity at the target (bounded so the pull-over stays inside the turn-radius gate);
        # gravity brings it down, and the sustaining motor cancels drag so speed holds into the dive.
        if boost_from_ground and horiz_dist < dive_range:
            desired = to_tgt / dist_tgt if dist_tgt > 1e-6 else vhat
            if float(np.dot(desired, vhat)) < 0.2:              # overflew the target -> dive to nadir, don't loop back
                desired = -up
            a_turn = (desired - vhat) * (speed / max(dive_tau, 1e-3))
            a_turn = a_turn - np.dot(a_turn, vhat) * vhat        # PURE turn (perpendicular) -> preserves airspeed
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
            # GRAVITY-TURN BOOST: while still climbing to the cruise band, thrust along an ALTITUDE-scheduled
            # pitch that rotates vertical -> target-heading, so the thrust stays ~ALONG velocity (a real rocket
            # launch, low lateral load) and the missile visibly RISES off the pad into the cruise band.
            if alt < cruise_alt_m and speed < cruise_speed_mps:
                frac = float(np.clip(alt / (0.7 * cruise_alt_m), 0.0, 1.0))   # pitch to horizontal by 70% cruise alt
                d = (1.0 - frac) * up + frac * head
                dhat = d / np.linalg.norm(d)
                return a_gravity + a_drag + dhat * boost_accel
            # CRUISE hold: level at cruise_alt/cruise_speed (P/D altitude + bounded forward thrust), minus
            # gravity/drag/centripetal so the HELD cruise is the ~zero-net-force regime a CV-EKF assumes.
            a_fwd = head * float(np.clip(0.05 * (cruise_speed_mps - speed), -60.0, 90.0))
            a_climb = up * (0.004 * (cruise_alt_m - alt) - 0.35 * vz)
            a_ctrl = -a_gravity - a_drag - up * (v_h ** 2 / r) + a_fwd + a_climb
        else:
            # LEGACY LEVEL CRUISE: lift cancels gravity, thrust cancels drag, MINUS the centripetal term the
            # vehicle needs to hold the curved level path -> net ~zero force (constant velocity), plus a gentle
            # heading correction toward the target so it stays on the great circle.
            a_ctrl = -a_gravity - a_drag - up * (v_h ** 2 / r)
            if nth > 1e-6 and t < weave_start_t:
                vh_vec = v - up * np.dot(v, up)
                nvh = np.linalg.norm(vh_vec)
                vhat_h = vh_vec / nvh if nvh > 1e-6 else head
                a_head = (head - vhat_h) * (speed * heading_gain)
                a_head = a_head - np.dot(a_head, vhat) * vhat        # PURE turn -> corrects heading without bleeding speed
                a_ctrl = a_ctrl + a_head

        # Terminal lateral S-weave, applied with the airframe ACTUATOR LAG so the micro-Doppler
        # anticipation cue (which reads the command) genuinely leads the velocity bend.
        a_applied = a_lat_applied(t)
        if a_applied != 0.0:
            cross = np.cross(up, vhat)
            ncross = np.linalg.norm(cross)
            if ncross > 1e-9:
                a_ctrl = a_ctrl + (cross / ncross) * a_applied
                # INDUCED DRAG: the terminal S-weave is a lift force -> it costs drag (~a_lat^2), so the hard
                # weave bleeds cruise speed instead of turning for free (energy-honest maneuvering).
                a_ctrl = a_ctrl + induced_drag_accel(v, abs(float(a_applied)))

        return a_gravity + a_drag + a_ctrl

    P, V = integrate(p0, v0, accel_fn, dt, n_steps)

    # The terminal dive ends at ground impact -- truncate there so the trajectory actually terminates on the
    # target instead of flying through the Earth or floating past.
    alts = np.array([altitude_of(P[i]) for i in range(len(P))])
    below = np.where(alts <= 0.0)[0]
    if len(below):
        n_steps = min(len(P), int(below[0]) + 1)
        P, V = P[:n_steps], V[:n_steps]
        
    dist_to_tgt = np.linalg.norm(P - target_surf, axis=1)
    dive_steps = np.where(dist_to_tgt < dive_range)[0]
    dive_start_step = int(dive_steps[0]) if len(dive_steps) > 0 else n_steps - 1

    # Record the COMMAND schedule (independent input) so the signature layer can drive the
    # anticipation cue from it; the kinematic response lags it by ACTUATOR_LAG_S.
    cmd_lat_accel = np.array([abs(a_lat_command(i * dt)) for i in range(n_steps)])

    meta = {
        "dt": dt,
        "n_steps": n_steps,
        "weave_start_frac": weave_start_frac,
        "weave_start_step": min(int(weave_start_t / dt), n_steps - 1),   # COMMAND onset (weave, before the dive)
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
