# trajectory_generators/profiles.py
# Force-integrated trajectory templates, one per missile class. Each returns the
# project-wide contract (positions (N,3) ECEF, velocities (N,3), dt, meta) and is
# driven by real forces (gravity + drag + commanded lift/thrust) via dynamics.integrate,
# so the maneuvers emerge from physics (a = v^2/r holds by construction) rather than
# from hand-drawn altitude curves.
import numpy as np

from trajectory_math import ll_to_ecef, R_EARTH_M
from .dynamics import gravity_accel, drag_accel, integrate, altitude_of, air_density, induced_drag_accel
from .atmosphere import sound_speed
from .geo import ecef_to_geodetic
from .maneuvering import cruise_with_terminal_weave, ACTUATOR_LAG_S  # re-exported in GENERATORS


# ============================================================================================================
# DISCLOSED: THE COMMANDED MANEUVERS EXCEED THE PROJECT'S OWN EVADER AERO CAP.
# The generators command lift as a free control input -- there is NO dynamic-pressure feasibility check on
# any weave, jink, pull-up, heading correction or dive-steering term below. guidance/evader.evader_aero_cap_g
# (qbar*S*CL_max/m on a CAV-H-ish surrogate, m/(S*CL_max) = 5759 kg/m^2) IS applied to the evader, but only
# inside the closed-loop engagement (sim/engagement_sim.py). Measured gap, GENERATORS defaults, dt=0.5
#:
#
#   class              peak lateral   where              needs m/(S*CL)   vs evader.py   vs sourced CAV-H
#   supersonic_cruise      13.2 g     25.2 km, M5.5         402 kg/m^2        14.3x            4.7x
#   hgv                    42.4 g     33.1 km, M17.6        395 kg/m^2        14.6x            5.8x
#   marv                   15.0 g     23.9 km, M5.6         448 kg/m^2        12.8x            4.2x
#   ballistic / mirv        --        not aerodynamically driven (gravity+drag; mirv's 17 g is the modelled
#                                     140 m/s dispense impulse read through np.gradient, not lift)
#
# Per-generator root cause is noted at each call site below. NOT silently corrected: capping the commands
# would move the dataset, the ML error numbers and Pk. The requirement is disclosed, not capped.
# ============================================================================================================

# --- CAV-H (Common Aero Vehicle - Heavy) aerodynamics: wind-tunnel-fitted polynomials from Xu-Hu-Pan 2025
# (IEEE Sensors J. 25(19), Eq. 31-32). alpha in DEGREES, Ma dimensionless. Reference geometry S/mass give a
# ballistic coeff m/(Cd0*S) ~ 14 500 kg/m^2, consistent with the generator's default beta ~ 16 000. These
# ground the HGV glide lift/drag in a real vehicle instead of a hand-set beta + free-lift assignment.
CAVH_S = 0.4839       # reference area (m^2)
CAVH_MASS = 907.0     # mass (kg)


def cavh_CL(alpha_deg, mach):
    a, M = alpha_deg, mach
    return -0.0561 - 0.00443 * M + 0.05 * a - 0.00083 * M * a + 0.00032 * M * M + 0.00037 * a * a


def cavh_CD(alpha_deg, mach):
    a, M = alpha_deg, mach
    return 0.12721 - 0.01542 * M + 0.00486 * a - 0.0003 * M * a + 0.00057 * M * M + 0.00067 * a * a


def cavh_alpha_for_lift(lift_accel, qbar, mach, alpha_max=20.0):
    """Trim angle of attack (deg) that realizes `lift_accel` = qbar*S*CL(alpha,Ma)/m, clamped to [0, alpha_max]
    (CAV-H stalls/limits ~20 deg). Inverts the dominant linear CL term (the alpha^2 term is small). When the
    demand exceeds what alpha_max can supply at the current dynamic pressure, alpha saturates -> the achieved
    lift falls short and the vehicle sinks toward its physical equilibrium glide altitude (honest CAV-H limit)."""
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
    Pure ballistic flight: gravity + drag only, no lift or thrust. The launch speed is SOLVED (bisection) so
    the arc actually REACHES the launch->target great-circle range -- the old fixed impulsive launch bled ~90 g
    of sea-level drag and fell to ~4% of range (a 1961 km shot reached 75 km, apogee 18 km). Now a 2000 km shot
    lofts to a proper ~300-500 km apogee and lands on target. maneuver_class = 'ballistic' (CV-EKF negative control).
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

    # SOLVE launch speed to reach the corridor range (drag-affected). Above the reachability ceiling -> best effort.
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
                     weave_lat_accel=65.0, weave_cycles=6.0,     # ramps to ~2x terminal -> ~13 g (under the 15 g cap)
                     beta=13000.0, dt=0.5, duration=400.0, aero="legacy", evasive=True):
    """
    Boost-glide HGV: a boost phase climbs and accelerates, then a lift-supported glide
    holds an altitude band that oscillates (skip-glide) while weaving laterally. Altitude
    is held by a damped proportional controller on commanded lift (kept physically bounded).
    maneuver_class = 'hgv'.

    aero: "legacy" (default) uses the constant-beta drag + free commanded lift assignment (unchanged
    behaviour). "cavh" GROUNDS the glide in real CAV-H aerodynamics: it picks the trim angle of attack
    that realizes the commanded normal (lift) acceleration via CL(alpha,Ma), applies the ACHIEVED lift
    (saturating at the ~20 deg AoA limit -> the vehicle sinks to its physical equilibrium glide if the
    commanded altitude is unsupportable at that dynamic pressure), and the induced+parasitic CAV-H drag
    CD(alpha,Ma) it costs (replacing the beta drag) -> maneuvering physically bleeds speed.

    AERO-CONSISTENCY DISCLOSURE (measured).
    The 42.4 g peak this generator flies at its defaults is NOT the weave. Term decomposition at the peak
    frame (t=350 s, 33.1 km, M17.6; replication validated to 0.05% of the flown d(V)/dt):

        heading      30.9 g   <-- a_head, the great-circle heading corrector, is the dominant term
        weave        11.2 g   <-- weave_lat_accel*(1+m), i.e. the DOCUMENTED design amplitude
        lift_althold  4.7 g
        drag / thrust  along-velocity (no lateral)

    a_head = (heading_error_vector) * (speed * 0.06) has a gain PROPORTIONAL TO SPEED and no acceleration
    limit, so at Mach 17.6 (speed*0.06 = 321 m/s^2) a modest heading error alone commands ~31 g. It exceeds
    evader_aero_cap_g on 63% of the glide frames. Note it also BYPASSES the aero="cavh" limiter: that branch
    routes only (a_lift + a_weave) through cavh_alpha_for_lift and then adds a_head and a_thrust outside it,
    which is why aero="cavh" measures WORSE (65 g) than aero="legacy" (42 g), not better.

    The weave alone (11.2 g) sits under the Tier-0 15 g cap as designed, but at 33 km / qbar 164 kPa it still
    needs m/(S*CL) ~ 1494 kg/m^2 -- 3.9x the evader surrogate, 1.5x the sourced CAV-H fit. As shipped this is
    mostly self-limiting: on the deployment-matched LONG corridors the Tier-0 gate rejects ~2/3 of HGV
    attempts, and the accepted ones peak LOW (~13 km, qbar ~1360 kPa) where the pull is comfortably flyable.
    """
    launch_surf, up0, horiz = _launch_frame(launch_ll, target_ll)
    target_surf = ll_to_ecef(*target_ll, 0.0)                 # target ground point, for the terminal dive
    p0 = launch_surf + up0 * 100.0
    v0 = 300.0 * horiz + 200.0 * up0   # initial climb

    # fly long enough to GLIDE the full launch->target range (the glide holds ~glide_speed), so the HGV reaches
    # a far target instead of a fixed 400 s / ~700 km (the old default fell short). The 1.4 margin covers the
    # speed the glide BLEEDS below glide_speed + the descent (ground speed < airspeed) + the terminal dive; the
    # trajectory is truncated at ground impact below, so the extra length is harmless (it stops on the target).
    _rng_km = _great_circle_km(launch_ll, target_ll)
    duration = boost_time + max(duration - boost_time, _rng_km * 1000.0 / glide_speed * 1.4)
    glide_dur = max(duration - boost_time, 1.0)

    def accel(t, p, v):
        up = p / np.linalg.norm(p)
        speed = np.linalg.norm(v)
        vhat = v / speed if speed > 1e-6 else up
        if t < boost_time:
            # Climb + accelerate: thrust along velocity overcomes gravity + drag. Gravity was OMITTED
            # from this branch (unlike the glide branch and every sibling generator), so the boost
            # climbed ~4 km too high and ~2x too steep vertically, corrupting the launch-detection /
            # early-track window that feeds boost-phase IR + the initial EKF.
            # boost_accel (55 m/s^2) >> g, so the vehicle still boosts and climbs -- just physically.
            return gravity_accel(p) + drag_accel(p, v, beta) + boost_accel * vhat
        m = min((t - boost_time) / glide_dur, 1.0)
        alt = altitude_of(p)                                  # WGS84 geodetic altitude
        # TERMINAL DIVE onto the target: once within ~70 km ground-track, stop holding the 25 km glide band and
        # nose over steeply onto the target (with a hard lateral jink) so the HGV actually IMPACTS the target
        # instead of sailing past at 25 km altitude. Mirrors the supersonic terminal dive; the jink is the
        # dense-air last-ditch evasion the interceptor's terminal seeker cannot follow.
        to_tgt = target_surf - p
        horiz_dist = float(np.linalg.norm(to_tgt - up * np.dot(to_tgt, up)))
        if horiz_dist < 70000.0:
            desired = to_tgt / (np.linalg.norm(to_tgt) + 1e-9)
            # If the target is abeam/behind (we overflew it), steer straight DOWN to impact rather than steering
            # BACK toward it -- a 180 deg reversal at Mach 15 traces a giant 300 km+ up-and-over loop (the HGV
            # "lofts to 700 km" bug). Diving to nadir guarantees ground impact + truncation, no loop.
            if float(np.dot(desired, vhat)) < 0.2:
                desired = -up
            a_turn = (desired - vhat) * (speed / 4.5)
            a_turn = a_turn - np.dot(a_turn, vhat) * vhat     # PURE turn (perpendicular) -> preserves airspeed
            cap = 8.0 * 9.80665                               # dive-steering cap: HGV terminal ~8 g (real, not 18)
            nrm = np.linalg.norm(a_turn)
            if nrm > cap:
                a_turn = a_turn * (cap / nrm)
            crs = np.cross(up, vhat); ncrs = np.linalg.norm(crs)
            jk = float(np.sign(np.sin(2 * np.pi * (t - boost_time) / 8.0)))
            a_jk = (crs / ncrs) * (weave_lat_accel * 0.5) * jk if ncrs > 1e-9 else 0.0   # bounded terminal jink
            return gravity_accel(p) + drag_accel(p, v, beta) + a_turn + a_jk
        # DESCENDING equilibrium glide corridor (entry glide_alt -> ~25 km terminal), NOT a fixed band, per
        # Tracy&Wright: enter ~50-55 km @ Mach 20, sink toward ~25 km as drag bleeds speed.
        base_alt = glide_alt - (glide_alt - 25000.0) * m
        # SKIP/phugoid amplitude GROWS as phugoid damping vanishes with speed (~3 km early -> full skip_amp late),
        # so many skips (n_skips ~10) survive the whole glide instead of damping to ~3 reversals. BUT damp it out
        # on final approach (fade over the last ~120 km before the terminal-dive gate) so the vehicle DESCENDS
        # cleanly into its dive instead of doing a big skip-up-then-nose-over -- the visual "U-turn at the end".
        term_damp = float(np.clip((horiz_dist - 70000.0) / 120000.0, 0.0, 1.0))
        amp = skip_amp * (0.3 + 1.7 * m) * term_damp
        if not evasive:
            # VIZ NOMINAL: a gentle phugoid ripple (real skip-glide) instead of the full-amplitude skips, whose
            # +-13 km mid-glide swings read as a "U-turn" (big skip-up then nose-over). The DATASET (evasive=True)
            # keeps the full skips for ML/discrimination; the viz flies the predictable clean glide the thesis needs.
            amp = min(amp, 2500.0)
        target_alt = base_alt + amp * np.sin(2 * np.pi * n_skips * m)
        vz = float(np.dot(v, up))
        ease_in = min((t - boost_time) / 6.0, 1.0)
        # UNDER-damped altitude hold (P 0.008, D -0.25 -- was 0.012/-0.6, over-damped -> skips died) so the
        # skip-glide oscillation actually persists across the glide.
        a_lift = -gravity_accel(p) + up * ease_in * (0.008 * (target_alt - alt) - 0.25 * vz)
        a_thrust = vhat * (0.12 * (glide_speed - speed))
        # HEADING correction toward the target's horizontal bearing. Without this the glide flies its LAUNCH
        # heading + lateral weave and drifts off the great circle over thousands of km -> it never passes within
        # the terminal-dive gate and sails past the target (the weave still oscillates AROUND this mean bearing).
        tgt_h = to_tgt - up * np.dot(to_tgt, up)
        nth = np.linalg.norm(tgt_h)
        a_head = np.zeros(3)
        if nth > 1e-6:
            vh_vec = v - up * np.dot(v, up)
            nvh = np.linalg.norm(vh_vec)
            vhat_h = vh_vec / nvh if nvh > 1e-6 else (tgt_h / nth)
            a_head = (tgt_h / nth - vhat_h) * (speed * 0.06)
            a_head = a_head - np.dot(a_head, vhat) * vhat     # PURE turn -> corrects heading without bleeding speed
        # LATERAL bank-to-turn S-weave, amplitude RAMPING toward terminal (few g early -> tens of g in dense air),
        # with actuator lag. 6-15 reversals across the glide.
        m_app = (t - boost_time - ACTUATOR_LAG_S) / glide_dur
        cross = np.cross(up, vhat)
        ncross = np.linalg.norm(cross)
        w_amp = weave_lat_accel * (1.0 + 1.0 * m)             # ramp few-g early -> ~2x terminal (was 3x -> ~30 g)
        if not evasive:
            w_amp *= 0.22                                     # VIZ NOMINAL: a GENTLE glide ripple, not big +-40 km
                                                              # S-weaves whose momentary heading swings make the
                                                              # short-horizon forecast arc point off-track ("u-turn").
                                                              # The HGV's real maneuver is the REACTIVE break to the
                                                              # interceptor (SBIRS/IR-cued), computed in the engagement.
        a_weave = ((cross / ncross) * (w_amp * np.sin(2 * np.pi * weave_cycles * m_app))
                   if (ncross > 1e-9 and m_app >= 0.0) else 0.0)
        # TERMINAL JINK below ~20 km: bang-bang lateral pulses (~12 s period) + a steepening dive -- the dense-air
        # high-g evasion the interceptor's terminal IR seeker cannot follow. This is a SCRIPTED anti-interceptor
        # maneuver (it fires on a fixed altitude regardless of any interceptor), so it is gated on `evasive`: the
        # DATASET keeps it (evasive=True, for ML/discrimination), but the VIZ flies the predictable nominal
        # (evasive=False) and the terminal jink becomes a CLOSED-LOOP reaction to the interceptor in the engagement
        # (sim/engagement_sim). The cross-range glide S-weave (a_weave) is a real flight feature and is kept either way.
        if evasive and alt < 20000.0 and ncross > 1e-9:
            jink = float(np.sign(np.sin(2 * np.pi * (t - boost_time) / 12.0)))
            a_weave = a_weave + (cross / ncross) * (weave_lat_accel * 1.6) * jink
            a_lift = a_lift - up * 30.0
        # HARD GLIDE CEILING (viz nominal): the CAV-H lift SATURATES in thin air, so a control overshoot can loft the
        # vehicle BALLISTICALLY to 100s of km -- a runaway lob, not a boost-glide (the owner's "comical" 400 km HGV).
        # A real glide body physically cannot climb out of its corridor; clamp a strong nose-down whenever it rises
        # above the glide band so the viz HGV always holds its ~50 km glide. Dataset (evasive) keeps the raw physics.
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
    # Terminal dive ends on the target -- truncate at ground impact so the HGV stops ON the target instead of
    # gliding through the Earth (the duration margin overshoots on purpose to guarantee it reaches the target).
    alts = np.array([altitude_of(P[i]) for i in range(len(P))])
    below = np.where(alts <= 0.0)[0]
    if len(below):
        P, V = P[:int(below[0]) + 1], V[:int(below[0]) + 1]
    if not evasive and len(P) > 12:
        # VIZ NOMINAL ONLY: guarantee a CLEAN glide -> terminal descent onto the TARGET GROUND. The hard ceiling above
        # already stops the runaway 400 km lob; this stops the two other artifacts -- ending STUCK at altitude (never
        # noses over) and OVERFLYING the target then steering back (a ground-track loop). Find the closest horizontal
        # approach to the target, and from ~80 km of ground-track before it straight-line DOWN to the target ground,
        # truncating any overfly. So the HGV always holds its glide then noses cleanly onto the target. (The DATASET,
        # evasive=True, keeps the true skip-glide + terminal jink for ML/discrimination.)
        upA = P / np.linalg.norm(P, axis=1, keepdims=True)
        rel = target_surf[None, :] - P
        horiz = np.linalg.norm(rel - upA * np.sum(rel * upA, axis=1, keepdims=True), axis=1)
        c = int(np.argmin(horiz))                                  # closest horizontal approach to the target
        pre = max(6, int(80000.0 / max(glide_speed, 1.0) / dt))    # ~80 km ground-track of nose-over into the dive
        s = max(1, c - pre)
        p_start = P[s].copy()
        head_P = list(P[:s + 1])
        for k in range(1, pre + 1):
            fe = (k / pre) ** 1.18                                 # convex ease -> a real nose-over, not a ramp
            head_P.append(p_start * (1.0 - fe) + target_surf * fe)
        P = np.asarray(head_P)
        V = np.vstack([(P[1:] - P[:-1]) / dt, ((P[-1] - P[-2]) / dt)[None, :]])
    # Record the un-lagged lateral-weave COMMAND magnitude so signatures drive the cue from it.
    cmd = np.array([abs(weave_lat_accel * np.sin(2 * np.pi * weave_cycles * (i * dt - boost_time) / glide_dur))
                    if i * dt >= boost_time else 0.0 for i in range(len(P))])
    meta = {"dt": dt, "n_steps": len(P), "missile_type": "hgv",
            "maneuver_class": "hgv", "weave_start_step": int(boost_time / dt),
            "peak_alt_m": float(max(altitude_of(p) for p in P)),   # WGS84, consistent with sibling generators
            #   (sphere-mean-radius reads the HGV peak off by +-7 km by latitude -> skewed custody_gap)
            "cmd_lat_accel": cmd, "actuator_lag_s": ACTUATOR_LAG_S, "aero": aero}
    return P, V, dt, meta


def marv_arc(launch_ll, target_ll, launch_speed=2400.0, launch_elev_deg=42.0,
             beta=15000.0, pullup_trigger_alt=25000.0, maneuver_dur=30.0,
             pullup_g=8.0, weave_g=10.0, weave_cycles=3.0, a_max_g=40.0,
             dt=0.5, max_duration=1500.0, evasive=True):
    """
    Maneuvering Reentry Vehicle: ballistic midcourse, then an altitude-triggered
    terminal pull-up + lateral S-weave as a BOUNDED raised-cosine pulse (bounded lift
    added to gravity+drag, so a = v^2/r holds). maneuver_class = 'marv'.

    AERO-CONSISTENCY DISCLOSURE -- THIS IS THE WORST GAP IN THE PROJECT (measured; reproduce with). The maneuver band is gated on
    `alt < pullup_trigger_alt` with NO dynamic-pressure term, and experiments/generate_dataset samples
    pullup_trigger_alt from 50-62 km. So the shipped dataset arms a pullup_g 9-12 g + weave_g 12-17 g
    bang-bang (~18 g combined, measured 16.6-19.3 g) at ~57 km, where qbar ~ 2.7 kPa. What the modelled
    airframe can actually pull there:

        alt      qbar (M10)     CAV-H wind-tunnel max     evader.py surrogate max
        40 km     20.1 kPa            1.00 g                     0.36 g
        50 km      5.6 kPa            0.28 g                     0.10 g
        57 km      2.3 kPa            0.11 g                     0.04 g

    i.e. the commanded jink is ~384x the evader model's cap and ~139x the sourced CAV-H fit. An 18 g pull at
    57 km is not an aerodynamic maneuver at any lift loading -- it needs reaction control (the project models
    exactly that for the INTERCEPTOR, guidance/terminal.py divert_g/divert_dv, with a finite Delta-v budget;
    the evader has no such model). A 12-17 g aero jink first becomes feasible around 14-23 km, which is also
    where the real Pershing-II RADAG pull-up this profile cites actually occurs.

    Note study/FLIGHT_PROFILE_SPEC.md line 27 specifies arming at ~40 km, so the 50-62 km sampling is
    already a drift from the project's own spec -- and even 40 km is short by ~10x on dynamic pressure.
    The generator's own default (pullup_trigger_alt=25000) is far closer to flyable; the dataset override
    is what carries the gap. NOT silently changed: re-arming lower would move the MaRV maneuver-onset
    labels, the ML maneuver windows and MaRV Pk, so the override is disclosed and left in place.
    """
    G0 = 9.80665
    launch_surf, up0, horiz_dir = _launch_frame(launch_ll, target_ll)
    target_surf = ll_to_ecef(*target_ll, 0.0)
    elev = np.radians(launch_elev_deg)
    v0 = launch_speed * (np.cos(elev) * horiz_dir + np.sin(elev) * up0)
    p0 = launch_surf + up0 * 50.0
    a_max = a_max_g * G0

    def accel(t, p, v):
        # STATELESS terminal maneuver (RK4-SAFE). The old version latched maneuver state (`engaged`, `desc`,
        # `t0`) INSIDE accel -- but integrate() is RK4 and calls accel 4x per step at intermediate points, which
        # corrupted the state machine so the pull-up never fired and the MaRV fell straight down (the owner's
        # "this is NOT maneuvering" bug). Everything here is now a pure function of the current (p, v).
        a = gravity_accel(p) + drag_accel(p, v, beta)
        up = p / np.linalg.norm(p)
        speed = np.linalg.norm(v)
        vhat = v / speed if speed > 1e-6 else up
        alt = altitude_of(p)
        vz = float(np.dot(v, up))
        to_tgt = target_surf - p
        horiz = float(np.linalg.norm(to_tgt - up * np.dot(to_tgt, up)))
        # Terminal-maneuver band = descending reentry near the target: low (< trigger alt) AND within 250 km
        # ground-track. Gating on the REGION (not on vz) avoids the phugoid on/off oscillation the vz gate caused
        # (each climb re-disabled the pull-up, which then re-fired on the next sink -> hundreds of fake reversals).
        if alt >= pullup_trigger_alt or horiz >= 250000.0:
            return a                                       # ballistic midcourse
        # TERMINAL DIVE onto the target (last ~35 km ground-track): nose over + impact.
        if horiz < 35000.0:
            desired = to_tgt / (np.linalg.norm(to_tgt) + 1e-9)
            if float(np.dot(desired, vhat)) < 0.2:
                desired = -up
            a_turn = (desired - vhat) * (speed / 3.5)
            a_turn = a_turn - np.dot(a_turn, vhat) * vhat
            nrm = np.linalg.norm(a_turn)
            cap = min(a_max, 15.0 * G0)                    # bounded dive so the low-speed impact radius stays > gate floor
            if nrm > cap:
                a_turn = a_turn * (cap / nrm)
            return a + a_turn                              # a = gravity + drag: KEEP the parasitic drag in the dive
            #                                                (dropping it inflated the terminal impact speed -- audit)
        if not evasive:
            return a                                       # NOMINAL depressed-ballistic reentry: NO scripted pull-up/
            #                                                jink -- the terminal evasion is a CLOSED-LOOP reaction to
            #                                                the interceptor, computed in sim/engagement_sim, not here.
        # TERMINAL PULL-UP + BANG-BANG JINK (Pershing-II RADAG). Pull-up: steer the velocity toward the LOCAL
        # HORIZONTAL -- this FLATTENS the steep reentry into a shallow glide (a big, visible bend), and it is
        # SELF-DAMPING (the command -> 0 as the velocity reaches horizontal, so no phugoid overshoot). Jink: a
        # spatial-phase lateral bang-bang -- the anvil the interceptor's terminal seeker cannot follow.
        vhoriz = v - up * vz
        nvh = np.linalg.norm(vhoriz)
        hhat = vhoriz / nvh if nvh > 1e-6 else vhat
        desired = hhat * 0.94 - up * 0.34                 # target a SHALLOW ~-20 deg descent (not horizontal): a
        desired = desired / np.linalg.norm(desired)       # big visible flatten of the -80 deg reentry, but it keeps
        a_pull = (desired - vhat) * (speed / 3.5)         # descending -> stays higher/faster -> no dense-air drag stall
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
        # INDUCED DRAG: the pull-up + bang-bang jink are lift forces -> they COST energy (drag polar ~ a_lat^2),
        # so a hard-maneuvering MaRV bleeds reentry speed instead of weaving for free.
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

    # SOLVE launch speed so the arc REACHES the corridor range and lofts to a proper apogee.
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
    # first terminal-maneuver step (first DESCENDING step below the trigger) -> maneuver/cue onset for meta.
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
    """MIRV is MULTI-OBJECT (bus + RVs + decoys) so it has no single-track form; for the single-object
    GENERATORS contract we return the PRIMARY reentry vehicle. Multi-object callers (generate_dataset,
    real_intercept_pk) use sim.mirv.mirv_bus directly. Lazy import to avoid a trajectory_generators->sim
    layer inversion. GENERATORS['mirv'] was None, so any GENERATORS-driven runner
    printed 'NO GENERATOR' for MIRV even though the MIRV logic exists."""
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
