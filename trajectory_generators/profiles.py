# trajectory_generators/profiles.py
# Force-integrated trajectory templates, one per missile class. Each returns the
# project-wide contract (positions (N,3) ECEF, velocities (N,3), dt, meta) and is
# driven by forces (gravity + drag + commanded lift/thrust) via dynamics.integrate,
# so maneuvers follow from the equations of motion and a = v^2/r holds by construction.
import numpy as np

from trajectory_math import ll_to_ecef, R_EARTH_M
from .dynamics import gravity_accel, drag_accel, integrate, altitude_of, air_density, induced_drag_accel
from .atmosphere import sound_speed
from .geo import ecef_to_geodetic
from .maneuvering import cruise_with_terminal_weave, ACTUATOR_LAG_S  # re-exported in GENERATORS


# Aerodynamic feasibility. The generators command lift as a free control input, with no dynamic-pressure
# feasibility check on any weave, jink, pull-up, heading correction or dive-steering term, and the commands
# are not capped. Peak lateral acceleration at the GENERATORS defaults (dt = 0.5 s) and the lift loading it
# requires, compared with a CAV-H-like evader surrogate (qbar*S*CL_max/m with m/(S*CL_max) = 5759 kg/m^2)
# and with the sourced CAV-H fit:
#
#   class              peak lateral   where              needs m/(S*CL)   vs surrogate   vs sourced CAV-H
#   supersonic_cruise      13.2 g     25.2 km, M5.5         402 kg/m^2        14.3x            4.7x
#   hgv                    42.4 g     33.1 km, M17.6        395 kg/m^2        14.6x            5.8x
#   marv                   15.0 g     23.9 km, M5.6         448 kg/m^2        12.8x            4.2x
#   ballistic / mirv        --        not aerodynamically driven (gravity + drag; mirv's 17 g is the modelled
#                                     140 m/s dispense impulse read through np.gradient)
#
# The hgv and marv docstrings break down the dominant terms.

# CAV-H (Common Aero Vehicle - Heavy) aerodynamics: wind-tunnel-fitted polynomials from Xu-Hu-Pan 2025
# (IEEE Sensors J. 25(19), Eqs. 31-32). alpha in degrees, Ma dimensionless. The reference area and mass give
# a ballistic coefficient m/(Cd0*S) ~ 14 500 kg/m^2, consistent with the generator's default beta ~ 16 000.
CAVH_S = 0.4839       # reference area (m^2)
CAVH_MASS = 907.0     # mass (kg)


def cavh_CL(alpha_deg, mach):
    a, M = alpha_deg, mach
    return -0.0561 - 0.00443 * M + 0.05 * a - 0.00083 * M * a + 0.00032 * M * M + 0.00037 * a * a


def cavh_CD(alpha_deg, mach):
    a, M = alpha_deg, mach
    return 0.12721 - 0.01542 * M + 0.00486 * a - 0.0003 * M * a + 0.00057 * M * M + 0.00067 * a * a


def cavh_alpha_for_lift(lift_accel, qbar, mach, alpha_max=20.0):
    """Trim angle of attack (deg) that realizes `lift_accel` = qbar*S*CL(alpha,Ma)/m, clamped to [0, alpha_max].

    Takes the positive root of the quadratic CL(alpha, Ma) = CL_req; the CAV-H limit is ~20 deg. When the
    demand exceeds what alpha_max supplies at the current dynamic pressure, alpha saturates, the achieved
    lift falls short, and the vehicle sinks toward its equilibrium glide altitude."""
    if qbar < 1e-3:
        return 0.0
    cl_req = lift_accel * CAVH_MASS / (qbar * CAVH_S)
    c0 = -0.0561 - 0.00443 * mach + 0.00032 * mach * mach       # CL at alpha=0
    c1 = 0.05 - 0.00083 * mach                                  # dCL/dalpha (linear)
    c2 = 0.00037                                                # d^2CL/dalpha^2 (quadratic)
    disc = c1 * c1 - 4.0 * c2 * (c0 - cl_req)                   # solve c2*a^2 + c1*a + (c0 - cl_req) = 0
    alpha = (-c1 + np.sqrt(disc)) / (2.0 * c2) if disc >= 0.0 else 0.0   # physical (+alpha) root
    return float(np.clip(alpha, 0.0, alpha_max))


def _geodetic_alts(P):
    """True (WGS84) altitude above the ellipsoid for each ECEF point, metres."""
    return np.array([ecef_to_geodetic(*p)[2] for p in P])


def _launch_frame(launch_ll, target_ll):
    """Return (launch_surface_ecef, up_unit, horizontal_unit_toward_target)."""
    launch_surf = ll_to_ecef(*launch_ll, 0.0)
    tgt_surf = ll_to_ecef(*target_ll, 0.0)
    up = launch_surf / np.linalg.norm(launch_surf)
    to_t = tgt_surf - launch_surf
    horiz = to_t - up * np.dot(to_t, up)
    horiz = horiz / np.linalg.norm(horiz)
    return launch_surf, up, horiz


def _great_circle_km(a, b):
    la1, lo1 = np.radians(a[0]), np.radians(a[1])
    la2, lo2 = np.radians(b[0]), np.radians(b[1])
    d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * (R_EARTH_M / 1000.0) * np.arcsin(np.sqrt(d))


def ballistic_arc(launch_ll, target_ll, launch_speed=2200.0, launch_elev_deg=42.0,
                  beta=12000.0, dt=0.5, max_duration=1800.0):
    """
    Ballistic flight under gravity and drag, with no lift or thrust. The launch speed is solved by bisection
    so the arc reaches the launch-target great-circle range; a 2000 km shot lofts to a ~300-500 km apogee
    and lands on the target. maneuver_class = 'ballistic' (CV-EKF negative control).
    """
    launch_surf, up, horiz = _launch_frame(launch_ll, target_ll)
    elev = np.radians(launch_elev_deg)
    p0 = launch_surf + up * 50.0

    def accel(t, p, v):
        return gravity_accel(p) + drag_accel(p, v, beta)

    def _fly(vmag):
        v0 = vmag * (np.cos(elev) * horiz + np.sin(elev) * up)
        n = int(max_duration / dt)
        P, V = integrate(p0, v0, accel, dt, n)
        alt = _geodetic_alts(P)
        hit = np.where(alt[2:] <= 0.0)[0]
        end = (hit[0] + 2) if len(hit) else n
        P, V, alt = P[:end], V[:end], alt[:end]
        imp = ecef_to_geodetic(*P[-1])[:2]
        return _great_circle_km(launch_ll, imp), P, V, alt

    # Solve the launch speed that reaches the corridor range with drag; fly at 16 km/s if even that falls short.
    rng_km = _great_circle_km(launch_ll, target_ll)
    lo_v, hi_v = 2000.0, 16000.0
    if _fly(hi_v)[0] < rng_km:
        vmag = hi_v
    else:
        for _ in range(22):
            mid = 0.5 * (lo_v + hi_v)
            if _fly(mid)[0] < rng_km:
                lo_v = mid
            else:
                hi_v = mid
        vmag = 0.5 * (lo_v + hi_v)
    _, P, V, alt = _fly(vmag)
    meta = {"dt": dt, "n_steps": len(P), "missile_type": "ballistic",
            "maneuver_class": "ballistic", "weave_start_step": None,
            "peak_alt_m": float(alt.max()), "launch_speed_solved": float(vmag)}
    return P, V, dt, meta


def boost_glide_skip(launch_ll, target_ll, boost_time=25.0, boost_accel=55.0,
                     glide_alt=55000.0, glide_speed=5500.0,
                     n_skips=10, skip_amp=8000.0,
                     weave_lat_accel=65.0, weave_cycles=6.0,     # ramps to ~2x at terminal, ~13 g (under the 15 g cap)
                     beta=13000.0, dt=0.5, duration=400.0, aero="legacy", evasive=True):
    """
    Boost-glide HGV. A boost phase climbs and accelerates, then a lift-supported glide follows a
    descending, oscillating altitude band (skip-glide) while weaving laterally, and a terminal dive ends
    on the target. Altitude is held by a damped proportional controller on commanded lift.
    maneuver_class = 'hgv'.

    aero: "legacy" (default) uses constant-beta drag and freely assigned commanded lift. "cavh" picks the
    trim angle of attack that realizes the commanded normal (lift) acceleration through CL(alpha, Ma),
    applies the achieved lift (saturating at the ~20 deg AoA limit, so the vehicle sinks to its equilibrium
    glide when the commanded altitude is unsupportable at that dynamic pressure), and replaces the beta
    drag with the induced plus parasitic CAV-H drag CD(alpha, Ma), so maneuvering bleeds speed.

    evasive=False flies a nominal path: reduced skip and weave amplitudes, no low-altitude terminal jink,
    a glide ceiling, and a smooth nose-over onto the target.

    At the defaults the lateral acceleration peaks at 42.4 g (t = 350 s, 33.1 km, M17.6). Terms at that frame:

        heading      30.9 g   a_head, the great-circle heading corrector (dominant)
        weave        11.2 g   weave_lat_accel*(1+m), the design amplitude
        lift_althold  4.7 g
        drag / thrust  along-velocity (no lateral component)

    a_head = (heading error vector) * (speed * 0.06) has a gain proportional to speed and no acceleration
    limit, so at Mach 17.6 (speed*0.06 = 321 m/s^2) a modest heading error commands ~31 g. It exceeds the
    evader surrogate's aerodynamic cap on 63% of glide frames. The aero="cavh" branch routes only
    (a_lift + a_weave) through cavh_alpha_for_lift and adds a_head and a_thrust outside it, so its peak is
    65 g against 42 g for aero="legacy".

    The weave alone (11.2 g) is under the 15 g lateral-load cap; at 33 km (qbar 164 kPa) it needs
    m/(S*CL) ~ 1494 kg/m^2, 3.9x the evader surrogate and 1.5x the sourced CAV-H fit. On the long
    corridors the physical-envelope checks reject ~2/3 of HGV attempts, and accepted trajectories peak low
    (~13 km, qbar ~1360 kPa), where the pull is aerodynamically flyable.
    """
    launch_surf, up0, horiz = _launch_frame(launch_ll, target_ll)
    target_surf = ll_to_ecef(*target_ll, 0.0)                 # target ground point, for the terminal dive
    p0 = launch_surf + up0 * 100.0
    v0 = 300.0 * horiz + 200.0 * up0   # initial climb

    # Duration long enough to glide the full launch-target range at ~glide_speed. The 1.4 margin covers speed
    # bled below glide_speed, the descent (ground speed < airspeed) and the terminal dive; the trajectory is
    # truncated at ground impact below, which discards the extra length.
    _rng_km = _great_circle_km(launch_ll, target_ll)
    duration = boost_time + max(duration - boost_time, _rng_km * 1000.0 / glide_speed * 1.4)
    glide_dur = max(duration - boost_time, 1.0)

    def accel(t, p, v):
        up = p / np.linalg.norm(p)
        speed = np.linalg.norm(v)
        vhat = v / speed if speed > 1e-6 else up
        if t < boost_time:
            # Boost: gravity, drag and thrust along velocity. boost_accel (55 m/s^2) exceeds g, so the
            # vehicle climbs and accelerates.
            return gravity_accel(p) + drag_accel(p, v, beta) + boost_accel * vhat
        m = min((t - boost_time) / glide_dur, 1.0)
        alt = altitude_of(p)                                  # WGS84 geodetic altitude
        # Terminal dive: within 70 km ground track, leave the glide band and nose over steeply onto the target
        # with a bounded lateral jink, as in the supersonic terminal dive.
        to_tgt = target_surf - p
        horiz_dist = float(np.linalg.norm(to_tgt - up * np.dot(to_tgt, up)))
        if horiz_dist < 70000.0:
            desired = to_tgt / (np.linalg.norm(to_tgt) + 1e-9)
            # If the target is abeam or behind, steer to nadir. A 180 deg reversal at Mach 15 would trace an
            # up-and-over loop of 300 km or more; diving to nadir guarantees ground impact and truncation.
            if float(np.dot(desired, vhat)) < 0.2:
                desired = -up
            a_turn = (desired - vhat) * (speed / 4.5)
            a_turn = a_turn - np.dot(a_turn, vhat) * vhat     # pure turn (perpendicular), preserves airspeed
            cap = 8.0 * 9.80665                               # dive-steering cap, ~8 g HGV terminal
            nrm = np.linalg.norm(a_turn)
            if nrm > cap:
                a_turn = a_turn * (cap / nrm)
            crs = np.cross(up, vhat); ncrs = np.linalg.norm(crs)
            jk = float(np.sign(np.sin(2 * np.pi * (t - boost_time) / 8.0)))
            a_jk = (crs / ncrs) * (weave_lat_accel * 0.5) * jk if ncrs > 1e-9 else 0.0   # bounded terminal jink
            return gravity_accel(p) + drag_accel(p, v, beta) + a_turn + a_jk
        # Descending equilibrium glide corridor from glide_alt at entry to 25 km at the end of the glide, after
        # Tracy & Wright: entry at ~50-55 km and Mach 20, sinking toward ~25 km as drag bleeds speed.
        base_alt = glide_alt - (glide_alt - 25000.0) * m
        # Skip (phugoid) amplitude grows from 0.3*skip_amp to 2*skip_amp over the glide as phugoid damping falls
        # with speed, so the skips persist across the glide. It fades to zero over the 120 km before the
        # terminal-dive boundary, so the vehicle descends smoothly into the dive.
        term_damp = float(np.clip((horiz_dist - 70000.0) / 120000.0, 0.0, 1.0))
        amp = skip_amp * (0.3 + 1.7 * m) * term_damp
        if not evasive:
            # Nominal path: skip amplitude capped at 2.5 km, a gentle phugoid ripple.
            amp = min(amp, 2500.0)
        target_alt = base_alt + amp * np.sin(2 * np.pi * n_skips * m)
        vz = float(np.dot(v, up))
        ease_in = min((t - boost_time) / 6.0, 1.0)
        # Under-damped altitude hold (P 0.008, D -0.25) so the skip-glide oscillation persists across the glide.
        a_lift = -gravity_accel(p) + up * ease_in * (0.008 * (target_alt - alt) - 0.25 * vz)
        a_thrust = vhat * (0.12 * (glide_speed - speed))
        # Heading correction toward the target's horizontal bearing. It holds the glide on the great circle over
        # thousands of km, so the vehicle reaches the terminal-dive boundary; the weave oscillates about this bearing.
        tgt_h = to_tgt - up * np.dot(to_tgt, up)
        nth = np.linalg.norm(tgt_h)
        a_head = np.zeros(3)
        if nth > 1e-6:
            vh_vec = v - up * np.dot(v, up)
            nvh = np.linalg.norm(vh_vec)
            vhat_h = vh_vec / nvh if nvh > 1e-6 else (tgt_h / nth)
            a_head = (tgt_h / nth - vhat_h) * (speed * 0.06)
            a_head = a_head - np.dot(a_head, vhat) * vhat     # pure turn, corrects heading without bleeding speed
        # Lateral bank-to-turn S-weave with actuator lag, 2*weave_cycles reversals across the glide.
        m_app = (t - boost_time - ACTUATOR_LAG_S) / glide_dur
        cross = np.cross(up, vhat)
        ncross = np.linalg.norm(cross)
        w_amp = weave_lat_accel * (1.0 + 1.0 * m)             # ramps from 1x at glide entry to 2x at the end
        if not evasive:
            w_amp *= 0.22                                     # nominal path: gentle glide ripple
        a_weave = ((cross / ncross) * (w_amp * np.sin(2 * np.pi * weave_cycles * m_app))
                   if (ncross > 1e-9 and m_app >= 0.0) else 0.0)
        # Terminal jink below 20 km: bang-bang lateral pulses of 1.6*weave_lat_accel with a 12 s period, plus a
        # steepening dive (30 m/s^2 downward). It is scripted on altitude and runs only with evasive=True. The
        # cross-range glide S-weave (a_weave) is present in both modes.
        if evasive and alt < 20000.0 and ncross > 1e-9:
            jink = float(np.sign(np.sin(2 * np.pi * (t - boost_time) / 12.0)))
            a_weave = a_weave + (cross / ncross) * (weave_lat_accel * 1.6) * jink
            a_lift = a_lift - up * 30.0
        # Glide ceiling, nominal path only. CAV-H lift saturates in thin air, so a control overshoot can loft the
        # vehicle ballistically to hundreds of km. Above glide_alt + 6 km while climbing, a nose-down acceleration
        # returns it to the glide band. evasive=True applies no ceiling.
        ceil = np.zeros(3)
        if (not evasive) and alt > glide_alt + 6000.0 and vz > 0.0:
            ceil = -up * (0.03 * (alt - glide_alt - 6000.0) + 2.5 * vz)
        if aero == "cavh":
            rho = air_density(alt)
            qbar = 0.5 * rho * speed * speed
            mach = speed / max(sound_speed(alt), 1e-3)
            a_norm = a_lift + a_weave                              # total commanded aero-normal accel (vector)
            L_cmd = float(np.linalg.norm(a_norm))
            alpha = cavh_alpha_for_lift(L_cmd, qbar, mach)
            nhat = a_norm / (L_cmd + 1e-9)
            L_ach = qbar * CAVH_S * cavh_CL(alpha, mach) / CAVH_MASS   # achieved lift (may saturate < L_cmd)
            D_ach = qbar * CAVH_S * cavh_CD(alpha, mach) / CAVH_MASS   # CAV-H drag replaces the beta drag
            return gravity_accel(p) + nhat * L_ach + a_thrust + a_head - vhat * D_ach + ceil
        return gravity_accel(p) + drag_accel(p, v, beta) + a_lift + a_thrust + a_weave + a_head + ceil

    n = int(duration / dt)
    P, V = integrate(p0, v0, accel, dt, n)
    # Truncate at ground impact; the duration margin above ensures the trajectory reaches the target first.
    alts = np.array([altitude_of(P[i]) for i in range(len(P))])
    below = np.where(alts <= 0.0)[0]
    if len(below):
        P, V = P[:int(below[0]) + 1], V[:int(below[0]) + 1]
    if not evasive and len(P) > 12:
        # Nominal path only: the samples from ~80 km of ground track before the closest horizontal approach to the
        # target onward are replaced by a convex nose-over onto the target ground point, which removes any
        # overflight. evasive=True keeps the integrated trajectory.
        upA = P / np.linalg.norm(P, axis=1, keepdims=True)
        rel = target_surf[None, :] - P
        horiz = np.linalg.norm(rel - upA * np.sum(rel * upA, axis=1, keepdims=True), axis=1)
        c = int(np.argmin(horiz))                                  # closest horizontal approach to the target
        pre = max(6, int(80000.0 / max(glide_speed, 1.0) / dt))    # ~80 km ground-track of nose-over into the dive
        s = max(1, c - pre)
        p_start = P[s].copy()
        head_P = list(P[:s + 1])
        for k in range(1, pre + 1):
            fe = (k / pre) ** 1.18                                 # convex ease gives a nose-over profile
            head_P.append(p_start * (1.0 - fe) + target_surf * fe)
        P = np.asarray(head_P)
        V = np.vstack([(P[1:] - P[:-1]) / dt, ((P[-1] - P[-2]) / dt)[None, :]])
    # Record the un-lagged lateral-weave command magnitude so signatures drive the cue from it.
    cmd = np.array([abs(weave_lat_accel * np.sin(2 * np.pi * weave_cycles * (i * dt - boost_time) / glide_dur))
                    if i * dt >= boost_time else 0.0 for i in range(len(P))])
    meta = {"dt": dt, "n_steps": len(P), "missile_type": "hgv",
            "maneuver_class": "hgv", "weave_start_step": int(boost_time / dt),
            "peak_alt_m": float(max(altitude_of(p) for p in P)),   # WGS84, consistent with sibling generators
            #   (a mean-radius sphere misplaces the peak by up to +-7 km depending on latitude)
            "cmd_lat_accel": cmd, "actuator_lag_s": ACTUATOR_LAG_S, "aero": aero}
    return P, V, dt, meta


def marv_arc(launch_ll, target_ll, launch_speed=2400.0, launch_elev_deg=42.0,
             beta=15000.0, pullup_trigger_alt=25000.0, maneuver_dur=30.0,
             pullup_g=8.0, weave_g=10.0, weave_cycles=3.0, a_max_g=40.0,
             dt=0.5, max_duration=1500.0, evasive=True):
    """
    Maneuvering reentry vehicle: ballistic midcourse, then an altitude-triggered terminal pull-up and
    lateral bang-bang jink (bounded lift added to gravity and drag, so a = v^2/r holds), ending in a
    terminal dive onto the target. maneuver_class = 'marv'.

    The maneuver band is gated on `alt < pullup_trigger_alt` with no dynamic-pressure term.
    experiments/generate_dataset samples pullup_trigger_alt from 50-62 km, so those trajectories arm a
    9-12 g pull-up and a 12-17 g jink (~18 g combined, measured 16.6-19.3 g) near 57 km, where
    qbar ~ 2.7 kPa. Maximum aerodynamic lift acceleration at those altitudes:

        alt      qbar (M10)     CAV-H wind-tunnel max     evader surrogate max
        40 km     20.1 kPa            1.00 g                     0.36 g
        50 km      5.6 kPa            0.28 g                     0.10 g
        57 km      2.3 kPa            0.11 g                     0.04 g

    The commanded jink is ~384x the evader surrogate's cap and ~139x the sourced CAV-H fit; an 18 g pull
    at 57 km would require reaction control, which the evader model does not include. A 12-17 g aerodynamic
    jink becomes feasible around 14-23 km, the altitude band of the Pershing-II RADAG pull-up this profile
    follows. The generator default, pullup_trigger_alt = 25000 m, lies just above that band.
    """
    G0 = 9.80665
    launch_surf, up0, horiz_dir = _launch_frame(launch_ll, target_ll)
    target_surf = ll_to_ecef(*target_ll, 0.0)
    elev = np.radians(launch_elev_deg)
    v0 = launch_speed * (np.cos(elev) * horiz_dir + np.sin(elev) * up0)
    p0 = launch_surf + up0 * 50.0
    a_max = a_max_g * G0

    def accel(t, p, v):
        # The terminal maneuver is stateless. integrate() is RK4 and evaluates accel four times per step at
        # intermediate points, so accel is a pure function of the current (p, v).
        a = gravity_accel(p) + drag_accel(p, v, beta)
        up = p / np.linalg.norm(p)
        speed = np.linalg.norm(v)
        vhat = v / speed if speed > 1e-6 else up
        alt = altitude_of(p)
        vz = float(np.dot(v, up))
        to_tgt = target_surf - p
        horiz = float(np.linalg.norm(to_tgt - up * np.dot(to_tgt, up)))
        # Terminal-maneuver band: below the trigger altitude and within 250 km ground track of the target. The
        # band depends on position only, so a phugoid climb does not switch the pull-up off and on.
        if alt >= pullup_trigger_alt or horiz >= 250000.0:
            return a                                       # ballistic midcourse
        # Terminal dive onto the target within 35 km ground track.
        if horiz < 35000.0:
            desired = to_tgt / (np.linalg.norm(to_tgt) + 1e-9)
            if float(np.dot(desired, vhat)) < 0.2:
                desired = -up
            a_turn = (desired - vhat) * (speed / 3.5)
            a_turn = a_turn - np.dot(a_turn, vhat) * vhat
            nrm = np.linalg.norm(a_turn)
            cap = min(a_max, 15.0 * G0)                    # bounded dive keeps the low-speed turn radius above the turn-radius floor
            if nrm > cap:
                a_turn = a_turn * (cap / nrm)
            return a + a_turn                              # a = gravity + drag; parasitic drag stays on in the dive
        if not evasive:
            return a                                       # nominal depressed-ballistic reentry, no scripted pull-up or jink
        # Terminal pull-up and bang-bang jink (Pershing-II RADAG). The pull-up steers the velocity toward a shallow
        # descent, flattening the steep reentry; the command falls to zero as the velocity
        # reaches that direction, so it is self-damping. The jink is a lateral bang-bang phased on ground-track
        # distance.
        vhoriz = v - up * vz
        nvh = np.linalg.norm(vhoriz)
        hhat = vhoriz / nvh if nvh > 1e-6 else vhat
        desired = hhat * 0.94 - up * 0.34                 # shallow ~-20 deg descent: flattens the ~-80 deg
        desired = desired / np.linalg.norm(desired)       # reentry while still descending, which keeps speed
        a_pull = (desired - vhat) * (speed / 3.5)         # and altitude and avoids a dense-air drag stall
        a_pull = a_pull - np.dot(a_pull, vhat) * vhat
        pn = np.linalg.norm(a_pull)
        if pn > pullup_g * G0:
            a_pull = a_pull * (pullup_g * G0 / pn)
        right = np.cross(vhat, up)
        nr = np.linalg.norm(right)
        right = right / nr if nr > 1e-9 else np.zeros(3)
        phase = (250000.0 - horiz) / 215000.0            # 0 at 250 km ground-track -> 1 at the 35 km dive gate
        jink = float(np.sign(np.sin(2 * np.pi * weave_cycles * phase)))
        a_jink = right * (weave_g * G0) * jink
        # Induced drag: the pull-up and jink are lift forces and cost energy (drag polar ~ a_lat^2), so a hard
        # maneuver bleeds reentry speed.
        a_ind = induced_drag_accel(v, float(np.linalg.norm(a_pull + a_jink)))
        return a + a_pull + a_jink + a_ind

    def _fly(vmag):
        v0m = vmag * (np.cos(elev) * horiz_dir + np.sin(elev) * up0)
        n = int(max_duration / dt)
        Pf, Vf = integrate(p0, v0m, accel, dt, n)
        altf = _geodetic_alts(Pf)
        hitf = np.where(altf[2:] <= 0.0)[0]
        endf = (hitf[0] + 2) if len(hitf) else n
        return Pf[:endf], Vf[:endf], altf[:endf]

    # Solve the launch speed so the arc reaches the corridor range.
    _rng_km = _great_circle_km(launch_ll, target_ll)
    _lo, _hi = 2000.0, 16000.0
    if _great_circle_km(launch_ll, ecef_to_geodetic(*_fly(_hi)[0][-1])[:2]) < _rng_km:
        _vsol = _hi
    else:
        for _ in range(20):
            _mid = 0.5 * (_lo + _hi)
            if _great_circle_km(launch_ll, ecef_to_geodetic(*_fly(_mid)[0][-1])[:2]) < _rng_km:
                _lo = _mid
            else:
                _hi = _mid
        _vsol = 0.5 * (_lo + _hi)
    P, V, alt = _fly(_vsol)
    # First terminal-maneuver step (first descending step below the trigger), recorded as the maneuver/cue onset.
    weave_start = None
    cmd = np.zeros(len(P))
    for _i in range(len(P)):
        _u = P[_i] / np.linalg.norm(P[_i])
        if (float(np.dot(V[_i], _u)) < 0.0) and (alt[_i] < pullup_trigger_alt):
            if weave_start is None:
                weave_start = _i
            cmd[_i] = pullup_g * G0        # commanded pull-up magnitude while maneuvering (signature cue driver)
    meta = {"dt": dt, "n_steps": len(P), "missile_type": "marv",
            "maneuver_class": "marv", "weave_start_step": weave_start,
            "peak_alt_m": float(alt.max()), "cmd_lat_accel": cmd, "actuator_lag_s": ACTUATOR_LAG_S}
    return P, V, dt, meta


def _cruise_wrap(launch_ll, target_ll, **kw):
    """Adapt cruise_with_terminal_weave to the (P,V,dt,meta) registry contract,
    adding missile_type / maneuver_class to meta."""
    P, V, dt, meta = cruise_with_terminal_weave(launch_ll=launch_ll, target_ll=target_ll, **kw)
    meta = {**meta, "missile_type": "supersonic_cruise", "maneuver_class": "evasive",
            "peak_alt_m": float(_geodetic_alts(P).max())}
    return P, V, dt, meta


# Registry: missile_type -> generator(launch_ll, target_ll, **params) -> (P,V,dt,meta)
def _mirv_wrap(launch_ll, target_ll, **kw):
    """Primary reentry vehicle of a MIRV, for the single-object GENERATORS contract.

    MIRV is multi-object (bus, RVs, decoys); multi-object callers such as generate_dataset use
    sim.mirv.mirv_bus directly. sim.mirv is imported lazily so trajectory_generators does not import sim
    at module load."""
    from sim.mirv import mirv_bus
    objs, dt, meta = mirv_bus(launch_ll, target_ll, **{k: v for k, v in kw.items()
                                                       if k in ("num_rv", "num_decoy", "rng", "sep_dv")})
    rvs = [o for o in objs if o["type"] == "rv"] or objs
    o = rvs[0]
    return o["P"], o["V"], dt, meta


GENERATORS = {
    "supersonic_cruise": _cruise_wrap,
    "ballistic": ballistic_arc,
    "hgv": boost_glide_skip,
    "marv": marv_arc,
    "mirv": _mirv_wrap,                              # multi-object -> primary RV for single-track callers
}
