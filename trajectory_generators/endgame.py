"""
trajectory_generators/endgame.py - a terminal homing engagement whose lateral commands are
produced by a guidance law, at a step fine enough to resolve them.

WHY THIS GENERATOR IS SEPARATE FROM THE OTHER FIVE. Every generator in this package integrates at
dt = 0.5 s (0.2 s for the cruise one) because it exists to populate a tracking dataset whose radar
scans arrive at that rate. The micro-Doppler onset question lives on a ~35 ms fin-to-onset budget,
which such a grid under-samples by ~14x: a command shorter than a sample is not representable, and
interpolating one onto a fine grid does not resolve its shape, it invents it. This file therefore
integrates the ENTIRE engagement at DT_G = 1e-4 s -- the same step sim.sixdof.PitchAirframe is
driven at elsewhere in this project, and 5x finer than the 0.5 ms radar slow-time sample -- so the
command, the fin history and the achieved state are all resolved without interpolation anywhere.

WHAT MAKES THE COMMAND SOURCED RATHER THAN AUTHORED. The missile's lateral acceleration command is
not a waveform. It is

    a_c(t) = N' * V_c(t) * lambda_dot_hat(t)

proportional navigation: N' the navigation ratio, V_c the closing velocity and lambda_dot the
line-of-sight rate, all three read from the instantaneous relative geometry. Zarchan (Science and
Global Security 8(1):99-124, 1999) states the law in these terms -- "with proportional navigation,
acceleration commands are issued which are proportional to the line-of-sight rate between the
missile and target" -- and reports it as the law "in use for more than four decades on most of the
world's operational homing missiles". Nobody types the command's amplitude, its duration or its
shape here; they fall out of closing geometry, target motion and the guidance loop.

THE TIMESCALE, AND WHERE IT COMES FROM. Three things set how fast the command moves, and all three
are properties of the engagement rather than choices:

  * SEEKER ACQUISITION. Before acquisition there is no guidance and the fin is at rest; at
    acquisition the loop closes and the command rises through the noise filter that smooths the
    seeker's line-of-sight-rate estimate. Zarchan works his examples at guidance-system time
    constants of 0.05, 0.1, 0.2 and 0.5 s, and derives minimum achievable values of 0.12 s and
    0.3 s for two specific radome-slope cases; he does not state a fielded range, and none is
    claimed here. What he does state is that "in endoatmospheric missiles the dominant portion of
    the total system time constant is usually associated with the flight control system". TAU_F is
    swept, not drawn from a literature band, and the axis reported is the MEASURED loop constant.

    THE ACQUISITION INSTANT IS AN INPUT, and that is the honest limit of this event. The command's
    amplitude and shape at acquisition are the guidance law's -- N'*V_c*lambda_dot at the geometry
    that happens to obtain -- but WHEN the loop closes is set by t_mid, not by the engagement. This
    is seeker-acquisition onset, not target-maneuver onset, and it must not be quoted as the
    latter. The event whose timing is fully emergent is the jink below.
  * TARGET MANEUVER. Zarchan: "the maximum acceleration to take out target maneuver will occur near
    intercept". The target's evasion is an INPUT to the engagement -- disclosed, and the one
    authored waveform here -- but the missile's response to it is not: that is the guidance law
    acting on the geometry the evasion produces.
  * TIME TO GO. The line-of-sight rate diverges as t_go -> 0, so in the final fraction of a second
    the command sweeps its whole range whatever else is happening.

The generator does not decide which of the three dominates. experiments/endgame_lead.py measures the
command durations the engagement actually produces and runs the letter's detector on them.

MODEL AND ITS CEILING, stated so the reduction is visible:

  * PLANAR. The engagement runs in the local horizontal plane at a fixed altitude. sim.sixdof's
    airframe is a pitch-channel model taking a signed scalar command, so a planar engagement uses it
    exactly as written; a three-dimensional command would have to be collapsed to a magnitude, and a
    direction reversal would then reach the airframe as a dip through zero rather than as a sign
    change. Gravity acts equally on both vehicles and cancels from the relative geometry that PN
    reads, so it is carried as a common out-of-plane bias and the vertical channel is not modeled.
  * FLAT LOCAL FRAME. Positions are mapped to ECEF through the local east/north basis at the anchor
    point. Over the few kilometres an endgame covers the curvature error is metres.
  * NO SEEKER NOISE. lambda_dot is measured exactly and then passed through the first-order noise
    filter, so TAU_F contributes its lag without a noise process behind it. Adding glint or
    range-independent noise would raise the commanded acceleration near intercept, not lower it.
  * THE AIRFRAME IS THE PROJECT'S. sim.sixdof.DEFAULT_AIRFRAME, unmodified -- a coasting tactical
    terminal stage. Its honesty ceiling is stated in that file and applies here unchanged.

CONTRACT. endgame_intercept() returns (P, V, dt, meta) like every entry in profiles.GENERATORS, with
the engagement decimated to `export_dt` for consumers written against a scan-rate grid. It is
deliberately NOT registered in GENERATORS: that registry feeds the tracking dataset through
corridors_for / _sample_params / SPEC_KEY, and a class added there changes that dataset. The full
fine-grid engagement, which is what the micro-Doppler experiments read, comes from engagement().

    python -m trajectory_generators.endgame
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_math import ll_to_ecef, LAUNCH_LAT, LAUNCH_LON              # noqa: E402
from trajectory_generators.atmosphere import density, sound_speed           # noqa: E402
from sim.sixdof import PitchAirframe, DEFAULT_AIRFRAME, a_max_at            # noqa: E402

G0 = 9.80665

DT_G = 1e-4          # engagement integration step (s). Matches the step PitchAirframe is driven at
                     # in experiments/multiclass_lead.py, and is 5x finer than the 0.5 ms radar
                     # slow-time sample, so no observable in this file is ever interpolated.

# --- guidance ------------------------------------------------------------------------------------
N_PRIME = 4.0        # navigation ratio. Zarchan reports that PN "requires three times the
                     # acceleration capability of the target", which is the N' = 3 asymptote. He
                     # gives no single fielded value; 4 is a conventional choice and is not
                     # attributed to him.
# TAU_F is the seeker/noise-filter lag ONLY. The values Zarchan works with are TOTAL guidance-system
# time constants, of which he says "in endoatmospheric missiles the dominant portion ... is usually
# associated with the flight control system". That portion is not an assumption here: sim.sixdof
# resolves the flight control system and this project measures its closed-loop response at
# T_63 = 0.179 s. Setting TAU_F to one of his totals would apply the flight-control lag twice, which
# is the double count experiments/literature_onset_test.py exists to exhibit. TAU_F carries only the
# remainder, and the quantity swept and reported is the MEASURED total from loop_time_constant().
#
# The reachable span is therefore bounded below by the airframe: at TAU_F = 0 the measured total is
# already 0.186 s, so nothing faster than that can be flown on this vehicle and the sweep says
# nothing about loops below it.
TAU_F_BAND = (0.0, 0.30)

T_GO_FLOOR = 0.20    # the engagement is not modeled inside this time to go (s). Proportional
                     # navigation is singular at intercept -- lambda_dot goes as 1/t_go^2 and the
                     # command reverses through the whole envelope in the last few tens of
                     # milliseconds -- and every reduction this file makes fails there together:
                     # a point mass has no length, the planar constant-altitude frame has no
                     # endgame roll, and the closing velocity that divides t_go passes through
                     # zero. At a ~1200 m/s closing speed this floor is ~240 m, inside the arming
                     # and fuzing interval of a real round. Nothing after it is measured.


def _local_basis(lat_deg, lon_deg, alt_m):
    """(anchor_ecef, east_unit, north_unit) for the local horizontal plane at (lat, lon, alt)."""
    anchor = ll_to_ecef(lat_deg, lon_deg, alt_m)
    up = anchor / np.linalg.norm(anchor)
    z_axis = np.array([0.0, 0.0, 1.0])
    east = np.cross(z_axis, up)
    east = east / np.linalg.norm(east)
    north = np.cross(up, east)
    return anchor, east, north


def sample_params(rng):
    """Engagement parameters for one trajectory.

    Every value below is an initial condition or a vehicle property. None of them is the shape,
    duration or amplitude of a command: those are outputs.
    """
    return dict(
        alt_m=float(rng.uniform(8000.0, 12000.0)),
        mach_m=float(rng.uniform(2.5, 3.5)),          # coasting terminal stage
        v_t=float(rng.uniform(250.0, 350.0)),         # aircraft-class target
        aspect_deg=float(rng.uniform(150.0, 210.0)),  # target heading relative to the missile,
        #                                               180 deg = head-on
        r0_m=float(rng.uniform(6000.0, 10000.0)),     # range at seeker acquisition
        heading_err_deg=float(rng.uniform(3.0, 10.0)),  # missile heading error at acquisition;
        #                                                 Zarchan works his example at 10 deg
        tau_f=float(rng.uniform(*TAU_F_BAND)),
        t_mid=float(rng.uniform(1.0, 1.5)),           # unguided coast before acquisition
        jink_g=float(rng.uniform(4.0, 7.0)),          # target evasion, Zarchan's "6 g target
        #                                               maneuver" -- an INPUT, see the module note
        jink_tgo=float(rng.uniform(1.2, 3.0)),        # time to go at which the target breaks
        jink_roll_s=float(rng.uniform(0.6, 1.0)),     # target roll-in, raised-cosine
        beta=float(rng.uniform(8000.0, 16000.0)),     # missile ballistic coefficient, kg/m^2
        midcourse_g=0.0,                              # see below; zero is the coasting default
        midcourse_period_s=4.0,
    )


# MIDCOURSE ACTIVITY is an axis, not a property of the engagement, and it defaults to zero.
# A missile that coasts unguided until seeker acquisition has a quiescent pre-command window
# because the engagement gave it one. A real round under midcourse guidance need not: it is
# correcting toward a predicted intercept point, so its fins are already moving when the seeker
# acquires. The letter's threshold is a maximum over that window, so pre-command fin motion raises
# the threshold the acquisition transient then has to clear. Setting midcourse_g > 0 puts a slow
# lateral command on the coast so the question can be asked instead of assumed. Like the target's
# evasion this is an authored input; unlike the acquisition command it is never the thing measured.


def engagement(rng=None, dt=DT_G, t_max=14.0, **overrides):
    """Fly one terminal engagement and return every resolved history it produced.

    Returned keys, all on the same dt grid:
        t          time from the start of the unguided coast
        a_cmd      the guidance command, m/s^2, signed, perpendicular to the missile velocity
        a_ach      the lateral acceleration the airframe achieved against it
        delta      the fin deflection, rad -- the observable the micro-Doppler cue is rendered from
        R          missile-to-target range, m
        t_go       R / closing velocity, s
        lam_dot    the true line-of-sight rate, rad/s
        a_tgt      the target's own lateral acceleration, m/s^2
        p_m, v_m   missile position and velocity in the local plane
    plus t_acq (seeker acquisition), t_jink (target break), t_end (closest approach) and the
    parameters the engagement was flown with.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    p = sample_params(rng)
    p.update(overrides)

    alt = p["alt_m"]
    a_snd = sound_speed(alt)
    v_m0 = p["mach_m"] * a_snd
    rho = density(alt)

    # --- initial geometry, set at acquisition and coasted backwards through the midcourse leg ----
    # Missile at the origin heading +x; target at range r0 on the +x axis, closing at `aspect_deg`.
    # The missile's velocity is offset from the collision course by heading_err_deg, which is the
    # error PN has to take out and which Zarchan identifies as the dominant demand at acquisition.
    r0 = p["r0_m"]
    psi_t = np.radians(p["aspect_deg"])
    v_t_vec = p["v_t"] * np.array([np.cos(psi_t), np.sin(psi_t)])
    p_t = np.array([r0, 0.0])

    # collision-course lead angle, then perturb it by the heading error
    #   sin(lead) = |v_t| sin(aspect) / |v_m|  for a constant-bearing intercept
    s_lead = np.clip(p["v_t"] * np.sin(psi_t) / v_m0, -1.0, 1.0)
    psi_m = np.arcsin(s_lead) + np.radians(p["heading_err_deg"])
    v_m_vec = v_m0 * np.array([np.cos(psi_m), np.sin(psi_m)])

    t_mid = p["t_mid"]
    p_m = -v_m_vec * t_mid                       # rewind the unguided coast
    p_t = p_t - v_t_vec * t_mid

    af = PitchAirframe(DEFAULT_AIRFRAME)
    a_max = a_max_at(alt, p["mach_m"])

    n = int(t_max / dt)
    t = np.arange(n) * dt
    a_cmd = np.zeros(n); a_ach = np.zeros(n); delta = np.zeros(n)
    R = np.zeros(n); tgo = np.zeros(n); lam_d = np.zeros(n); a_tgt = np.zeros(n)
    P_m = np.zeros((n, 2)); V_m = np.zeros((n, 2))

    lam_dot_hat = 0.0
    t_jink = None
    i_end = n - 1
    R_prev = np.inf

    for i in range(n):
        ti = i * dt
        r = p_t - p_m
        Rr = float(np.linalg.norm(r))
        v_rel = v_t_vec - v_m_vec
        V_c = -float(np.dot(r, v_rel)) / max(Rr, 1e-6)
        lam_dot = float(r[0] * v_rel[1] - r[1] * v_rel[0]) / max(Rr * Rr, 1e-6)
        t_go = Rr / V_c if V_c > 1.0 else float("inf")

        # Stop at the modeling floor or at closest approach, BEFORE writing this sample: a sample
        # whose kinematics are recorded but whose command and fin are not would appear to the
        # detector as a step to zero, and that step is an artifact of where the loop ends.
        if ti > t_mid and (t_go <= T_GO_FLOOR or Rr > R_prev):
            i_end = i - 1
            break
        R_prev = Rr

        P_m[i] = p_m; V_m[i] = v_m_vec
        R[i] = Rr; tgo[i] = t_go; lam_d[i] = lam_dot

        # --- guidance -----------------------------------------------------------------------
        if ti >= t_mid:
            if p["tau_f"] > dt:
                lam_dot_hat += (lam_dot - lam_dot_hat) * (dt / p["tau_f"])
            else:
                lam_dot_hat = lam_dot        # a perfect seeker: the airframe is the only lag left
            cmd = N_PRIME * V_c * lam_dot_hat
        else:
            cmd = 0.0                              # unguided coast: no seeker, no PN command
        # The midcourse term runs THROUGH acquisition rather than switching off at it. Switching it
        # off would put a step in the command at exactly the instant being timed, so the control
        # would add a second transient at the same moment as the one it is supposed to leave alone.
        # Carried through, it changes only what the pre-command window contains, which is the
        # variable this axis exists to move. It is identically zero at the default.
        if p["midcourse_g"] > 0.0:
            cmd += p["midcourse_g"] * G0 * np.sin(2.0 * np.pi * ti / p["midcourse_period_s"])
        a_cmd[i] = cmd

        # --- missile: command -> airframe -> achieved lateral acceleration --------------------
        v_m_mag = float(np.linalg.norm(v_m_vec))
        ach = af.step(cmd, v_m_mag, alt, dt)
        a_ach[i] = ach
        delta[i] = af.delta

        v_hat = v_m_vec / max(v_m_mag, 1e-6)
        n_hat = np.array([-v_hat[1], v_hat[0]])    # left normal: +lam_dot turns the missile left
        a_drag = -0.5 * rho * v_m_mag * v_m_mag / p["beta"] * v_hat
        v_m_vec = v_m_vec + (ach * n_hat + a_drag) * dt
        p_m = p_m + v_m_vec * dt

        # --- target: straight until it breaks, then a bounded turn ----------------------------
        if t_jink is None and ti >= t_mid and t_go <= p["jink_tgo"]:
            t_jink = ti
        if t_jink is not None:
            m = np.clip((ti - t_jink) / p["jink_roll_s"], 0.0, 1.0)
            a_t = p["jink_g"] * G0 * 0.5 * (1.0 - np.cos(np.pi * m))
        else:
            a_t = 0.0
        a_tgt[i] = a_t
        vt_mag = float(np.linalg.norm(v_t_vec))
        vt_hat = v_t_vec / max(vt_mag, 1e-6)
        nt_hat = np.array([-vt_hat[1], vt_hat[0]])
        v_t_vec = v_t_vec + a_t * nt_hat * dt
        p_t = p_t + v_t_vec * dt

    # ZERO-EFFORT MISS at the modeling floor: the separation that would remain if neither vehicle
    # accelerated again. Range at the floor is ~V_c * T_GO_FLOOR by construction and says nothing
    # about whether the guidance closed; ZEM does.
    r_f = p_t - p_m
    v_f = v_t_vec - v_m_vec
    tgo_f = float(np.linalg.norm(r_f)) / max(-float(np.dot(r_f, v_f)) / max(np.linalg.norm(r_f), 1e-6), 1e-6)
    zem_vec = r_f + v_f * tgo_f
    zem = float(np.linalg.norm(zem_vec))

    sl = slice(0, i_end + 1)
    return dict(t=t[sl], a_cmd=a_cmd[sl], a_ach=a_ach[sl], delta=delta[sl], R=R[sl],
                t_go=tgo[sl], lam_dot=lam_d[sl], a_tgt=a_tgt[sl], p_m=P_m[sl], v_m=V_m[sl],
                dt=dt, alt=alt, V=v_m0, mach=p["mach_m"], a_max=a_max,
                t_acq=t_mid, t_jink=t_jink, t_end=float(t[i_end]),
                R_floor_m=float(R[i_end]), zem_m=zem, params=p)


def loop_time_constant(tau_f, alt=10000.0, mach=3.0, a_step=20.0, t_max=3.0, dt=DT_G):
    """MEASURE the total guidance-system time constant: seeker filter in series with the airframe.

    Zarchan's 0.05-0.5 s is a property of the whole loop, so the sweep axis has to be the whole
    loop. This steps the demand, passes it through the same first-order filter the engagement uses,
    drives the same airframe, and returns (T_63, T_10_90) of the ACHIEVED lateral acceleration.
    """
    af = PitchAirframe(DEFAULT_AIRFRAME)
    V = mach * sound_speed(alt)
    n = int(t_max / dt)
    az = np.zeros(n)
    hat = 0.0
    for i in range(n):
        if tau_f > dt:
            hat += (a_step - hat) * (dt / tau_f)
        else:
            hat = a_step
        az[i] = af.step(hat, V, alt, dt)
    ss = float(np.mean(az[-int(0.2 / dt):]))
    if ss <= 0:
        return float("nan"), float("nan")
    t = np.arange(n) * dt

    def first(frac):
        k = np.where(az >= frac * ss)[0]
        return float(t[k[0]]) if len(k) else float("nan")

    return first(0.632), first(0.9) - first(0.1)


def endgame_intercept(launch_ll=(LAUNCH_LAT, LAUNCH_LON), target_ll=None,
                      rng=None, export_dt=0.1, **kw):
    """The profiles.GENERATORS contract: (positions (N,3) ECEF m, velocities (N,3), dt s, meta).

    The engagement is flown at DT_G and decimated to export_dt, because the tier-0 audit and every
    other consumer of this contract differentiates the position track twice; at 1e-4 s that
    amplifies quantisation into meaningless jerk. The fine histories the micro-Doppler experiments
    read are not on this path -- call engagement() for those.
    """
    eng = engagement(rng=rng, **kw)
    k = max(1, int(round(export_dt / eng["dt"])))
    idx = np.arange(0, len(eng["t"]), k)
    anchor, east, north = _local_basis(launch_ll[0], launch_ll[1], eng["alt"])
    P = anchor[None, :] + eng["p_m"][idx, 0:1] * east[None, :] + eng["p_m"][idx, 1:2] * north[None, :]
    V = eng["v_m"][idx, 0:1] * east[None, :] + eng["v_m"][idx, 1:2] * north[None, :]
    meta = dict(missile_type="endgame", maneuver_class="homing",
                cmd_lat_accel=eng["a_cmd"][idx].tolist(),
                alt_m=eng["alt"], mach=eng["mach"], a_max=eng["a_max"],
                t_acq=eng["t_acq"], t_jink=eng["t_jink"], zem_m=eng["zem_m"],
                params=eng["params"])
    return P, V, float(k * eng["dt"]), meta


# ------------------------------------------------------------------------------------------------
# Self-test. Three properties are asserted, and each one is a way the generator could fail to be
# what it claims: the commands must be inside the airframe's own authority, they must be sub-second,
# and the flown track must survive the project's tier-0 trajectory audit.
# ------------------------------------------------------------------------------------------------
ONSET_G_MS2 = 2.0    # the project's departure-from-cruise threshold (experiments/class_profiles.py),
                     # restated here only so this file's event definition is the same one.


def _departure(t, x, i_pre0, i_pre1, thr=None, need_s=0.005):
    """First sustained departure of x from the trend it was on before i_pre1.

    A constant baseline is the right reference only for a vehicle that was doing nothing. At
    acquisition it is: the coast commands identically zero, so the fitted line is zero and this
    reduces to thresholding |x|. At the target's break it is not: the missile is already answering
    its heading error, on a smoothly decaying trajectory, and thresholding |x| against a constant
    would place the departure wherever that decay happened to cross -- before the break as often as
    after it. Fitting the line and thresholding the residual asks the same question in both cases.

    Returns (t_departure, index, (intercept, slope)); t_departure is None if x never departs.
    """
    thr = ONSET_G_MS2 if thr is None else thr
    if i_pre1 - i_pre0 < 8:
        c = (float(x[i_pre0]) if i_pre1 > i_pre0 else 0.0, 0.0)
    else:
        c1, c0 = np.polyfit(t[i_pre0:i_pre1], x[i_pre0:i_pre1], 1)
        c = (float(c0), float(c1))
    resid = np.abs(x - (c[0] + c[1] * t))
    need = max(2, int(need_s / (t[1] - t[0])))
    ab = resid > thr
    for i in range(i_pre1, len(ab) - need):
        if ab[i:i + need].all():
            return float(t[i]), int(i), c
    return None, -1, c


def events(eng, pre_s=0.60, post_s=2.00):
    """The two onsets an endgame produces, each as a window the detector can be run on.

    'acquisition' -- the seeker closes the loop on a vehicle that has been coasting unguided, so
        the pre-command window is quiescent by the engagement's own structure. This is the endgame's
        analogue of the letter's commanded step.
    'jink'        -- the missile answers the target's break. The pre-command window here holds the
        decaying heading-error takeout, so it is NOT quiescent, and that is the point: it is the
        case the letter's calibration bound is about.

    `t_cmd` is when the MISSILE's command departed, not when the event that provoked it happened.
    For acquisition the two coincide. For the jink they do not: the target breaks, the line-of-sight
    rate takes time to answer, and a budget measured from the target's break would be the sum of the
    target's response and the missile's. The letter's budget is fin-to-onset on one vehicle.

    EVERY WINDOW IS CLIPPED WHERE THE COMMAND LEAVES THE AIRFRAME'S AUTHORITY. Proportional
    navigation diverges as t_go -> 0, and the jink arrives late enough that a fixed post-window
    reaches into that divergence: measured over a 2 s post-window, a fifth of the jink's samples
    command more than a_max. A window containing saturation measures control authority, and a slow
    command and a saturated one fail to convert for different reasons. The clip is by the airframe's
    own a_max at the instantaneous speed, and an event clipped before its command peaks is marked
    `truncated`, so its rise is read as a lower bound rather than as a value.

    Each event also carries the pre-window trend of the achieved acceleration (`ach_trend`), so a
    consumer can apply the same departure rule to the response.
    """
    t, dt = eng["t"], eng["dt"]
    af = PitchAirframe(DEFAULT_AIRFRAME)
    spd = np.linalg.norm(eng["v_m"], axis=1)
    amax = np.array([af.a_max_at(eng["alt"], v) for v in spd])
    over = np.abs(eng["a_cmd"]) > amax
    out = []
    for kind, t0 in (("acquisition", eng["t_acq"]), ("jink", eng["t_jink"])):
        if t0 is None or t0 - pre_s < t[0] or t0 + 0.10 > t[-1]:
            continue
        i0 = int(round((t0 - pre_s) / dt))
        i1 = min(len(t) - 1, int(round((t0 + post_s) / dt)))
        j0 = int(round(t0 / dt))
        bad = np.where(over[j0:i1 + 1])[0]
        if len(bad):
            i1 = j0 + int(bad[0]) - 1
        # The detector needs the threshold window and the search span around the onset; an event
        # whose usable window cannot hold them is dropped rather than measured on a stub.
        if i1 - j0 < int(0.35 / dt):
            continue
        tw = t[i0:i1 + 1]
        cw, aw, dw = eng["a_cmd"][i0:i1 + 1], eng["a_ach"][i0:i1 + 1], eng["delta"][i0:i1 + 1]
        k0, k1 = 0, j0 - i0
        t_cmd, i_cmd, cmd_trend = _departure(tw, cw, k0, k1)
        if t_cmd is None:
            continue
        _t_ach, _i_ach, ach_trend = _departure(tw, aw, k0, k1)
        out.append(dict(kind=kind, i0=i0, i1=i1, i_pre=k1, i_cmd=i_cmd, t_cmd=t_cmd,
                        t_event=float(t0), cmd_trend=cmd_trend, ach_trend=ach_trend,
                        t=tw, a_cmd=cw, a_ach=aw, delta=dw,
                        t_go=eng["t_go"][i0:i1 + 1], alt=eng["alt"], V=eng["V"],
                        mach=eng["mach"], a_max=eng["a_max"]))
    return out


def command_timescale(ev):
    """How fast this event's command moves, in measured quantities only.

        t_rise  time from the command's departure from its pre-window trend (by ONSET_G_MS2) to the
                largest excursion that follows
        d_rate  peak |d(delta)/dt| over the event, rad/s. The statistic is a trailing mean of
                |d(phase)/dt|^2 and phase goes as sin(delta), so peak fin rate is what sets the
                detectable excursion. Two commands of the same nominal duration and different
                amplitude are not the same signal, and this is the quantity that separates them.

    NO ANALYTIC MAPPING ONTO THE LETTER'S DURATION AXIS IS DONE HERE. A sinusoid of period T rises
    to its peak in T/4, so 4*t_rise looks like the matching period, but t_rise is measured from a
    2 m/s^2 departure while the sinusoid's quarter-period is measured from zero -- the two clocks
    start at different places and the ratio is not 4. experiments/endgame_lead.py instead runs this
    same function over the sinusoids the published sweep actually uses and interpolates, so the axes
    are related by measurement.

    `truncated` marks an event whose command was still climbing where the model stops, so its rise
    is a LOWER bound.
    """
    t = np.asarray(ev["t"], float)
    c0, c1 = ev["cmd_trend"]
    d = np.abs(np.asarray(ev["a_cmd"], float) - (c0 + c1 * t))
    dd = np.abs(np.diff(np.asarray(ev["delta"], float))) / (t[1] - t[0])
    d_rate = float(dd.max()) if len(dd) else float("nan")

    i_on = int(ev["i_cmd"])
    if i_on < 0:
        return dict(t_rise=float("nan"), d_rate=d_rate, amp=float(d.max()), truncated=False)
    pk = i_on + int(np.argmax(d[i_on:]))
    return dict(t_rise=float(t[pk] - t[i_on]), d_rate=d_rate, amp=float(d[pk]),
                truncated=bool(pk >= len(d) - 2))


def _validate(n=8, verbose=True):
    from audit_dataset import audit_trajectory

    af = PitchAirframe(DEFAULT_AIRFRAME)
    checks = []
    ev_over, tail_over, tgo_over, misses, quiet = [], [], [], [], []
    ts = {"acquisition": [], "jink": []}
    for s in range(n):
        eng = engagement(rng=np.random.default_rng(50000 + s))
        g = eng["t"] >= eng["t_acq"]
        misses.append(eng["zem_m"])
        quiet.append(float(np.abs(eng["a_cmd"][~g]).max()))
        # a_max at the INSTANTANEOUS speed, not the initial one: the missile is coasting and slows,
        # so a fixed a_max taken at launch Mach would overstate the authority it still has.
        spd = np.linalg.norm(eng["v_m"], axis=1)
        amax = np.array([af.a_max_at(eng["alt"], v) for v in spd])
        over = np.abs(eng["a_cmd"]) > amax
        evs = events(eng)
        for ev in evs:
            ts[ev["kind"]].append(command_timescale(ev))
            sl = slice(ev["i0"], ev["i1"] + 1)
            ev_over.append(float(np.mean(over[sl])))
        tail = over & g
        tail_over.append(float(np.mean(tail)))
        if tail.any():
            tgo_over.append((float(eng["t_go"][tail].min()), float(eng["t_go"][tail].max())))

    acq = np.array([e["t_rise"] for e in ts["acquisition"]], float)
    jnk = np.array([e["t_rise"] for e in ts["jink"]], float)

    # (1) COMMANDS INSIDE THE AIRFRAME'S OWN AUTHORITY, OVER THE WINDOWS THAT ARE MEASURED.
    #     a_max is not a knob; it is qbar*Sref*CN_max/m for the project's airframe at the flown
    #     condition, and a command past it produces saturation, which is control authority and not
    #     an onset. The assertion is scoped to the event windows because that is where a measurement
    #     is taken. Proportional navigation is singular at intercept and does exceed authority in
    #     the run-in to the modeling floor; that is reported, with the interval it occupies, rather
    #     than averaged into a number that reads as though the events were saturating.
    f_ev = float(np.max(ev_over)) if ev_over else 0.0
    f_tail = float(np.mean(tail_over))
    lo = min((a for a, _b in tgo_over), default=float("nan"))
    hi = max((b for _a, b in tgo_over), default=float("nan"))
    checks.append(("commanded acceleration within a_max across every measured event window",
                   f_ev == 0.0, "worst event window %.2f%% over a_max" % (100 * f_ev)))
    checks.append(("any exceedance is confined to the terminal run-in",
                   not np.isfinite(hi) or hi <= 1.0,
                   "%.2f%% of guided samples, all at t_go %.2f-%.2f s (floor %.2f); the "
                   "acquisition event sits at t_go 8-10 s" % (100 * f_tail, lo, hi, T_GO_FLOOR)))

    # (2) THE PRE-ACQUISITION LEG IS QUIESCENT. The unguided coast is what gives the acquisition
    #     event a pre-command window; if the coast carried a command the event would not be an
    #     onset and the threshold rule would have nothing quiet to calibrate on.
    q = float(np.max(quiet))
    checks.append(("unguided coast commands nothing", q < 1e-12,
                   "peak |a_cmd| before acquisition = %.2e m/s^2" % q))

    # (3) THE COMMAND TIMESCALE TRACKS THE LOOP, NOT THE DRAW. The generator's claim is that the
    #     command duration is a consequence of the guidance-system time constant, so a fast loop
    #     must produce a fast command and a slow loop a slow one. Asserting a fixed sub-second
    #     median instead would assert the draw, and the draw is swept in
    #     experiments/endgame_command.py.
    fast = engagement(rng=np.random.default_rng(50000), tau_f=0.0)
    slow = engagement(rng=np.random.default_rng(50000), tau_f=0.30)
    tf = command_timescale(events(fast)[0])["t_rise"]
    tsw = command_timescale(events(slow)[0])["t_rise"]
    checks.append(("acquisition command speeds up when the guidance loop does", tf < tsw,
                   "rise %.3f s at tau_f=0 vs %.3f s at tau_f=0.30, same trajectory" % (tf, tsw)))
    checks.append(("acquisition rise spread reported over the drawn band", True,
                   "median %.3f s (range %.3f-%.3f, n=%d)"
                   % (np.median(acq), acq.min(), acq.max(), acq.size)))
    if jnk.size:
        checks.append(("jink rise reported -- the fully emergent event, see the module note", True,
                       "median %.3f s (range %.3f-%.3f, n=%d)"
                       % (np.median(jnk), jnk.min(), jnk.max(), jnk.size)))

    # (4) THE MEASURED LOOP CONSTANT IS THE AXIS, AND ITS FLOOR IS THE AIRFRAME. tau_f = 0 leaves
    #     the airframe alone, so the sweep cannot reach below the project's own measured T_63 and
    #     says nothing about faster loops. The assertion is that the axis MOVES with tau_f --
    #     an axis that did not would make every row the same experiment.
    t63_0, _ = loop_time_constant(0.0)
    t63_hi, _ = loop_time_constant(TAU_F_BAND[1])
    checks.append(("measured loop T_63 moves with the filter and floors at the airframe",
                   t63_hi > t63_0 > 0.15,
                   "T_63 %.3f s at tau_f=0 (airframe alone, the floor) to %.3f s at tau_f=%.2f"
                   % (t63_0, t63_hi, TAU_F_BAND[1])))

    # (5) THE FLOWN TRACK SURVIVES TIER-0. The 'supersonic' spec is the closest envelope in
    #     CLASS_SPECS by altitude and speed; its g_cap and turn-radius floor are written for a
    #     cruise missile and do not bound an interceptor, so they are reported and not asserted --
    #     check (1) is the physically correct bound on this vehicle.
    from audit_dataset import velocities_from, JERK_CAP
    P, V, dt_e, _meta = endgame_intercept(rng=np.random.default_rng(50000))
    passed, det = audit_trajectory(P, dt_e, "supersonic")
    core = [k for k in ("finite", "altitude", "speed") if k in det]
    core_ok = all(det[k][0] for k in core)
    checks.append(("tier-0 finite/altitude/speed on the exported track",
                   core_ok, "; ".join("%s %s" % (k, det[k][1]) for k in core)))

    #     The tier-0 jerk check reads a triple finite difference of the decimated track, and the
    #     engagement is truncated at the modeling floor while still turning hard, so its last
    #     samples carry a one-sided differencing artifact. That is a claim, so it is measured:
    #     the peak must sit in the final samples and the interior must be inside the cap.
    jj = np.linalg.norm(velocities_from(velocities_from(velocities_from(P, dt_e), dt_e), dt_e),
                        axis=1)
    checks.append(("tier-0 jerk peak is a truncation-boundary artifact, interior inside the cap",
                   int(np.argmax(jj)) >= len(jj) - 3 and float(jj[:-3].max()) <= JERK_CAP,
                   "peak %.0f m/s^3 at sample %d of %d; interior peak %.0f (cap %.0f)"
                   % (jj.max(), int(np.argmax(jj)), len(jj), jj[:-3].max(), JERK_CAP)))

    # (6) THE ENGAGEMENT ACTUALLY CLOSES. A guidance law that does not is not producing endgame
    #     commands, whatever their duration. Range at the modeling floor is ~V_c*T_GO_FLOOR by
    #     construction and would pass this check for a missile flying straight past, so the
    #     quantity is the zero-effort miss: what would remain with no further acceleration.
    m = float(np.median(misses))
    checks.append(("median zero-effort miss at the modeling floor under 10 m", m < 10.0,
                   "median %.2f m (max %.2f) at t_go = %.2f s" % (m, max(misses), T_GO_FLOOR)))

    if verbose:
        for name, ok, detail in checks:
            print("  %s  %-62s  %s" % ("PASS" if ok else "FAIL", name, detail))
        print("  [tier-0 full: %s] %s" % ("pass" if passed else "partial",
                                          "; ".join("%s=%s" % (k, v[1]) for k, v in det.items()
                                                    if not v[0]) or "all checks pass"))
    return all(c[1] for c in checks), checks


if __name__ == "__main__":
    print("ENDGAME HOMING GENERATOR -- self-test")
    ok, _ = _validate()
    print("\n%s" % ("PASS: the commands are the guidance law's, they track the loop constant, and "
                    "every\n      measured window is inside the airframe's authority. Whether they "
                    "are fast enough\n      to convert is a measurement, not a property of this "
                    "file -- see experiments/endgame_lead.py."
                    if ok else "FAIL: see above."))
    sys.exit(0 if ok else 1)
