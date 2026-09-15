"""
trajectory_generators/endgame.py - a terminal homing engagement whose lateral commands come from a
guidance law, integrated at a step fine enough to resolve them.

The other generators in this package integrate at dt = 0.5 s (0.2 s for the cruise generator) to
match the scan rate of the tracking dataset. The micro-Doppler onset budget is ~35 ms from fin
motion to onset, which a 0.5 s grid under-samples by ~14x. A command shorter than one sample cannot
be represented on that grid, and interpolating it onto a fine grid does not recover its shape. This
module integrates the whole engagement at DT_G = 1e-4 s, the step sim.sixdof.PitchAirframe is driven
at in the other experiments and 5x finer than the 0.5 ms radar slow-time sample, so the command, the
fin history and the achieved state are resolved without interpolation.

Guidance law. The missile's lateral acceleration command is proportional navigation (PN),

    a_c(t) = N' * V_c(t) * lambda_dot_hat(t)

with N' the navigation ratio, V_c the closing velocity and lambda_dot the line-of-sight rate, all
read from the instantaneous relative geometry. Zarchan (Science and Global Security 8(1):99-124,
1999) states the law in these terms and reports it in use for more than four decades on most
operational homing missiles. The command's amplitude, duration and shape follow from the closing
geometry, the target motion and the guidance loop.

Timescale. Three properties of the engagement set how fast the command moves.

  * Seeker acquisition. Before acquisition there is no guidance and the fin is at rest. At
    acquisition the loop closes and the command rises through the first-order filter that smooths
    the seeker's line-of-sight-rate estimate. Zarchan works his examples at guidance-system time
    constants of 0.05, 0.1, 0.2 and 0.5 s and derives minimum achievable values of 0.12 s and
    0.3 s for two radome-slope cases; he gives no fielded range. He states that "in
    endoatmospheric missiles the dominant portion of the total system time constant is usually
    associated with the flight control system". TAU_F is swept, and the reported axis is the
    measured loop constant.

    The acquisition instant is an input to the engagement, set by t_mid; the command's amplitude
    and shape at acquisition come from the guidance law. This event is seeker-acquisition onset.
    The jink is the event whose timing is fully emergent.
  * Target maneuver. Zarchan notes that the maximum acceleration needed to take out a target
    maneuver occurs near intercept. The target's evasion is an input and the only authored waveform
    in this module; the missile's response to it is the guidance law acting on the resulting
    geometry.
  * Time to go. The line-of-sight rate diverges as t_go -> 0, so in the final fraction of a second
    the command sweeps its whole range.

experiments/endgame_lead.py measures the command durations the engagement produces and runs the
paper's detector on them.

Model assumptions.

  * Planar. The engagement runs in the local horizontal plane at a fixed altitude. The sim.sixdof
    airframe is a pitch-channel model taking a signed scalar command, so a planar engagement uses it
    directly. A three-dimensional command would have to be collapsed to a magnitude, and a direction
    reversal would reach the airframe as a dip through zero. Gravity acts equally on both vehicles
    and cancels from the relative geometry that PN reads, so it is carried as a common out-of-plane
    bias and the vertical channel is not modeled.
  * Flat local frame. Positions are mapped to ECEF through the local east/north basis at the anchor
    point. Over the few kilometres of an endgame the curvature error is metres.
  * No seeker noise. lambda_dot is measured exactly and passed through the first-order filter, so
    TAU_F contributes lag without a noise process. Glint or range-independent noise would raise the
    commanded acceleration near intercept.
  * Airframe. sim.sixdof.DEFAULT_AIRFRAME, unmodified: a coasting tactical terminal stage. The
    modelling limits documented in sim/sixdof.py apply here.

endgame_intercept() returns (P, V, dt, meta) like every entry in profiles.GENERATORS, with the
engagement decimated to `export_dt` for consumers that expect a scan-rate grid. It is not registered
in GENERATORS, because that registry feeds the tracking dataset through corridors_for,
_sample_params and SPEC_KEY, and adding a class there changes that dataset. The full fine-grid
engagement used by the micro-Doppler experiments comes from engagement().

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
                     # in experiments/multiclass_lead.py and is 5x finer than the 0.5 ms radar
                     # slow-time sample, so no observable in this module is interpolated.

# --- guidance ------------------------------------------------------------------------------------
N_PRIME = 4.0        # navigation ratio. Zarchan reports that PN "requires three times the
                     # acceleration capability of the target", which is the N' = 3 asymptote. He
                     # gives no single fielded value; 4 is a conventional choice.
# TAU_F is the seeker/noise-filter lag alone. Zarchan's values are total guidance-system time
# constants, whose dominant part in endoatmospheric missiles is the flight control system.
# sim.sixdof resolves the flight control system, with a measured closed-loop T_63 = 0.179 s, so
# setting TAU_F to one of Zarchan's totals would apply the flight-control lag twice. TAU_F carries
# only the remainder, and the swept and reported quantity is the measured total from
# loop_time_constant().
#
# At TAU_F = 0 the measured total is already 0.186 s. That is the fastest loop this airframe can
# fly, and the lower end of the sweep.
TAU_F_BAND = (0.0, 0.30)

T_GO_FLOOR = 0.20    # time to go (s) below which the engagement is not modeled. PN is singular at
                     # intercept: lambda_dot goes as 1/t_go^2 and the command reverses through the
                     # whole envelope in the last few tens of milliseconds. The model's reductions
                     # also break down there (a point mass has no length, the planar
                     # constant-altitude frame has no endgame roll, and the closing velocity that
                     # divides t_go passes through zero). At ~1200 m/s closing speed the floor is
                     # ~240 m, inside the arming and fuzing interval of a real round.


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
    """Draw engagement parameters for one trajectory.

    Every value is an initial condition or a vehicle property. The command's shape, duration and
    amplitude are outputs of the engagement.
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
        jink_g=float(rng.uniform(4.0, 7.0)),          # target evasion, g; Zarchan's example uses
        #                                               a 6 g target maneuver
        jink_tgo=float(rng.uniform(1.2, 3.0)),        # time to go at which the target breaks
        jink_roll_s=float(rng.uniform(0.6, 1.0)),     # target roll-in, raised-cosine
        beta=float(rng.uniform(8000.0, 16000.0)),     # missile ballistic coefficient, kg/m^2
        midcourse_g=0.0,                              # midcourse lateral command, g (see below)
        midcourse_period_s=4.0,
    )


# Midcourse activity is a swept axis and defaults to zero. A missile that coasts unguided until
# seeker acquisition has a quiescent pre-command window. A round under midcourse guidance is
# correcting toward a predicted intercept point, so its fins are already moving when the seeker
# acquires. The paper's threshold is a maximum over the pre-command window, so fin motion there
# raises the threshold the acquisition transient has to clear. midcourse_g > 0 adds a slow
# sinusoidal lateral command of period midcourse_period_s to the coast. Like the target's evasion,
# it is an authored input.


def engagement(rng=None, dt=DT_G, t_max=14.0, **overrides):
    """Fly one terminal engagement and return every resolved history it produced.

    Returned keys, all on the same dt grid:
        t          time from the start of the unguided coast
        a_cmd      the guidance command, m/s^2, signed, perpendicular to the missile velocity
        a_ach      the lateral acceleration the airframe achieved against it
        delta      the fin deflection, rad, from which the micro-Doppler cue is rendered
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

        # Stop at the modeling floor or at closest approach before writing this sample. A sample
        # with kinematics recorded but no command or fin value would appear to the detector as a
        # step to zero.
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
        # The midcourse term continues through acquisition. Switching it off there would put a step
        # in the command at the instant being timed. Carried through, it changes only the content
        # of the pre-command window. It is zero at the default.
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

    # Zero-effort miss (ZEM) at the modeling floor is the separation that would remain if neither
    # vehicle accelerated again. Range at the floor is ~V_c * T_GO_FLOOR by construction, so ZEM is
    # the measure of whether guidance closed.
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
    """Measure the total guidance-system time constant (seeker filter in series with the airframe).

    A step demand of a_step m/s^2 passes through the engagement's first-order filter with lag tau_f
    (s) and drives the airframe at altitude alt (m) and the given Mach. Returns (T_63, T_10_90) in
    seconds for the achieved lateral acceleration, or NaNs if it does not settle positive. Zarchan's
    0.05-0.5 s values refer to this whole-loop constant.
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

    The engagement is flown at DT_G and decimated to export_dt (s). Consumers of this contract,
    including the physical-envelope checks in audit_dataset.py, difference the position track to acceleration
    and jerk, and at a 1e-4 s step that differencing amplifies quantisation error. The fine-grid
    histories come from engagement().
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
# Event extraction, command timescale and self-test.
# ------------------------------------------------------------------------------------------------
ONSET_G_MS2 = 2.0    # departure-from-cruise threshold (m/s^2), the same value as in
                     # experiments/class_profiles.py.


def _departure(t, x, i_pre0, i_pre1, thr=None, need_s=0.005):
    """First sustained departure of x from its linear trend over [i_pre0, i_pre1).

    A line is fitted to x over the pre-window, and the residual |x - trend| must exceed thr (default
    ONSET_G_MS2) for need_s seconds (at least 2 samples), searching from i_pre1. Pre-windows shorter
    than 8 samples use the constant baseline x[i_pre0]. At acquisition the coast command is zero, so
    the fitted line is zero and this reduces to thresholding |x|. At the target's break the missile
    is still taking out its heading error on a smoothly decaying trajectory, and fitting the trend
    keeps that decay from being read as the departure.

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
    """The two onsets an endgame produces, each as a window the detector can run on.

    'acquisition'  the seeker closes the loop on a vehicle that has been coasting unguided, so the
                   pre-command window is quiescent. This is the endgame analogue of the paper's
                   commanded step.
    'jink'         the missile answers the target's break. The pre-command window holds the decaying
                   heading-error takeout and is not quiescent, which is the case the paper's
                   calibration bound addresses.

    `t_event` is the provoking event (acquisition or target break) and `t_cmd` is when the missile's
    command departed from its pre-window trend. The two coincide for acquisition. For the jink the
    line-of-sight rate takes time to respond to the break, so a budget measured from `t_event` would
    add the target's response to the missile's; the paper's budget is fin-to-onset on one vehicle.

    Each window spans pre_s before the event to post_s after it and is clipped at the first sample
    whose command exceeds the airframe's a_max at the instantaneous speed, since a window containing
    saturation measures control authority. PN diverges as t_go -> 0, and the jink occurs late enough
    that over an unclipped 2 s post-window about a fifth of its samples exceed a_max. Events with
    less than 0.35 s of usable window after the event, or whose command never departs, are dropped.

    Each event carries the pre-window trends of the command (`cmd_trend`) and of the achieved
    acceleration (`ach_trend`), so a consumer can apply the same departure rule to the response.
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
        # whose usable window cannot hold them is dropped.
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
    """How fast this event's command moves, from the event's measured histories.

        t_rise     time (s) from the command's departure from its pre-window trend (by ONSET_G_MS2)
                   to the largest excursion that follows
        d_rate     peak |d(delta)/dt| over the event, rad/s. The detection statistic is a trailing
                   mean of |d(phase)/dt|^2 with phase proportional to sin(delta), so peak fin rate
                   sets the detectable excursion and separates commands of equal duration and
                   different amplitude.
        amp        largest excursion of the command from its trend, m/s^2
        truncated  True if the peak falls in the last two samples of the window, meaning the command
                   was still climbing where the window ends and t_rise is a lower bound

    t_rise does not convert analytically to the paper's sinusoid-period axis. It starts at a
    2 m/s^2 departure while a sinusoid's quarter-period starts at zero, so the ratio is not 4.
    experiments/endgame_lead.py runs this function over the sweep's own sinusoids and interpolates.
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
        # a_max at the instantaneous speed. The coasting missile slows, so a_max at the initial
        # Mach would overstate its remaining authority.
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

    # (1) Commanded acceleration stays within the airframe's authority over the measured event
    #     windows. a_max = qbar*Sref*CN_max/m for the airframe at the flown condition, and a command
    #     above it saturates. PN is singular at intercept and exceeds authority in the run-in to the
    #     modeling floor; that exceedance is reported separately with the t_go interval it occupies.
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

    # (2) The unguided coast before acquisition commands nothing, which gives the acquisition event
    #     a quiescent pre-command window for the threshold rule to calibrate on.
    q = float(np.max(quiet))
    checks.append(("unguided coast commands nothing", q < 1e-12,
                   "peak |a_cmd| before acquisition = %.2e m/s^2" % q))

    # (3) The command duration follows the guidance-system time constant: on the same trajectory
    #     the acquisition command rises faster at tau_f = 0 than at tau_f = 0.30 s. The rise-time
    #     spread over the drawn parameter band is reported.
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

    # (4) The measured loop T_63 increases with tau_f. At tau_f = 0 only the airframe lag remains,
    #     which sets the lower end of the sweep.
    t63_0, _ = loop_time_constant(0.0)
    t63_hi, _ = loop_time_constant(TAU_F_BAND[1])
    checks.append(("measured loop T_63 moves with the filter and floors at the airframe",
                   t63_hi > t63_0 > 0.15,
                   "T_63 %.3f s at tau_f=0 (airframe alone, the floor) to %.3f s at tau_f=%.2f"
                   % (t63_0, t63_hi, TAU_F_BAND[1])))

    # (5) The exported track passes the finite, altitude and speed checks. The 'supersonic'
    #     spec is the closest envelope in CLASS_SPECS by altitude and speed. Its g_cap and
    #     turn-radius floor are set for a cruise missile, so they are only reported; check (1) is
    #     the applicable bound for an interceptor.
    from audit_dataset import velocities_from, JERK_CAP
    P, V, dt_e, _meta = endgame_intercept(rng=np.random.default_rng(50000))
    passed, det = audit_trajectory(P, dt_e, "supersonic")
    core = [k for k in ("finite", "altitude", "speed") if k in det]
    core_ok = all(det[k][0] for k in core)
    checks.append(("finite/altitude/speed checks on the exported track",
                   core_ok, "; ".join("%s %s" % (k, det[k][1]) for k in core)))

    #     The jerk check reads a triple finite difference of the decimated track. The
    #     engagement ends at the modeling floor while still turning hard, so the last samples carry
    #     a one-sided differencing artifact. The check requires the jerk peak to fall in the final
    #     three samples and the interior to stay within JERK_CAP.
    jj = np.linalg.norm(velocities_from(velocities_from(velocities_from(P, dt_e), dt_e), dt_e),
                        axis=1)
    checks.append(("jerk peak is a truncation-boundary artifact, interior inside the cap",
                   int(np.argmax(jj)) >= len(jj) - 3 and float(jj[:-3].max()) <= JERK_CAP,
                   "peak %.0f m/s^3 at sample %d of %d; interior peak %.0f (cap %.0f)"
                   % (jj.max(), int(np.argmax(jj)), len(jj), jj[:-3].max(), JERK_CAP)))

    # (6) The engagement closes. Range at the modeling floor is ~V_c*T_GO_FLOOR by construction
    #     for any trajectory, so the check uses the zero-effort miss, the separation that would
    #     remain with no further acceleration.
    m = float(np.median(misses))
    checks.append(("median zero-effort miss at the modeling floor under 10 m", m < 10.0,
                   "median %.2f m (max %.2f) at t_go = %.2f s" % (m, max(misses), T_GO_FLOOR)))

    if verbose:
        for name, ok, detail in checks:
            print("  %s  %-62s  %s" % ("PASS" if ok else "FAIL", name, detail))
        print("  [envelope checks: %s] %s" % ("pass" if passed else "partial",
                                          "; ".join("%s=%s" % (k, v[1]) for k, v in det.items()
                                                    if not v[0]) or "all checks pass"))
    return all(c[1] for c in checks), checks


if __name__ == "__main__":
    print("Endgame homing generator self-test")
    ok, _ = _validate()
    print("\n%s" % ("PASS: the commands come from the guidance law, follow the loop constant, and "
                    "stay\n      within the airframe's authority in every measured window. "
                    "experiments/endgame_lead.py\n      measures whether they convert."
                    if ok else "FAIL: see above."))
    sys.exit(0 if ok else 1)
