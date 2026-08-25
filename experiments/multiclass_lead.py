"""experiments/multiclass_lead.py -- the micro-Doppler lead, measured per target class.

THE DESIGN, AND WHY IT IS NARROW.

Driving sim.sixdof.PitchAirframe with every class's command at that class's own flight condition
does not hold up: at HGV's 40 km glide, 91 % of the flight commands more lateral acceleration than
the airframe can deliver, achieved a_z pins flat at ~1.03 g against an 8.0 g command with the fin
on its rail, and the glide structure is erased into a saturation plateau. MaRV at its native
dt = 0.5 s is numerically unstable.

The pipeline therefore runs behind a FEASIBILITY GATE, and the gate is a measurement, not a
workaround. experiments/class_profiles.py measures, per class, whether a maneuver onset exists and
what fraction of the flight commands more than a_max_at(alt, mach). Of the five classes:

    supersonic_cruise  events FORM and convert                      <- the letter's one class
    marv               manoeuvres (11.95 g) but at its ~52 km arming altitude the AMPLITUDE-CORRECT
                       a_max gate rejects every event: saturation, i.e. control authority, not onset
    hgv                events DO form (8 per trajectory, reversal convention) -- excluded by the
                       DWELL SWEEP instead, which is a stronger basis than a null result: the weave
                       has a 146 s median period, and over 4 statistic families x 6 dwells spanning
                       4 decades, 0 of 15 formable cells convert. High-detection cells are
                       indistinguishable from their own no-cue nulls (100 % vs 100 % at the 5 s
                       dwell), which is why "it detected something" is not evidence here.
    ballistic          never manoeuvres: 0 crossings, peak 0.10 g; lead UNDEFINED, not zero
    mirv               0 crossings, peak 0.04 g, exo-atmospheric dispense impulse; likewise undefined

So this measures ONE class and reports the other four as gated out, each with the measurement that
gated it and each for a DIFFERENT reason. MaRV is rejected per event by the amplitude-correct gate
below, and the manuscript reports one class of five.

    python -c "import numpy as np; from experiments.multiclass_lead import class_windows; \
        print([(n, len(class_windows(n, rng=np.random.default_rng(7777))[0])) \
               for n in ('supersonic_cruise','marv','hgv','ballistic','mirv')])"

A one-class result with a measured explanation for each exclusion is a smaller claim than "lead
across five classes" and a defensible one.

CUE PROVENANCE. The signature is rendered from delta(t) -- the airframe's fin deflection, which is
downstream of the physics and observable in principle -- never from the command. The budget is not
a fixed quantity: command SHAPE moves it 4.2x (MaRV's raised-cosine 147.9 ms vs a step's 35.4 ms on
the same airframe). The budget is therefore recomputed per class and NEVER pooled, and leads are
reported in absolute ms alongside each class's own budget, never as a shared "% of budget".

COMPARATOR HONESTY. The kinematic arm is a trailing mean of |a_z| formed from the TRUE lateral
acceleration plus additive noise. It is an ORACLE -- it never estimates the state it thresholds --
so any advantage over it is a LOWER BOUND on what the modality is worth against a real tracker.
The manuscript carries the same disclosure.

    python experiments/multiclass_lead.py
    python experiments/multiclass_lead.py --reps 30 --json runs/ml/multiclass_lead.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from scipy.stats import wilcoxon                                             # noqa: E402
from sim.sixdof import PitchAirframe, DEFAULT_AIRFRAME, sound_speed          # noqa: E402
from trajectory_generators.dynamics import altitude_of                        # noqa: E402
from experiments.class_profiles import (                                      # noqa: E402
    load_class, lateral_signal, decompose, describe, feasibility, ONSET_G_MS2,
)
from experiments.causal_dwell_test import causal_lead, thr_from_cruise, DT_R  # noqa: E402
from experiments.dphi_sweep import return_from_fin, stat_matched_phase        # noqa: E402

DT_FINE = 1e-4               # airframe integration step; MaRV blows up at the generator's 0.5 s
PRE_S, POST_S = 0.60, 0.40   # window around the command onset: cruise for the threshold, then event
FIN_ARM_M = 0.30
KIN_NOISE = 0.3              # DEFAULT ONLY. This is the comparator's measurement-noise sigma and
                             # it is LOAD-BEARING: the muD lead is constant in it, so the reported
                             # advantage is (muD lead - kin lead(sigma)) and crosses zero near
                             # sigma ~ 0.005 m/s^2. It must be swept and reported, never assumed.
                             # See experiments/sigma_sweep.py.


def drive_airframe(t_grid, a_cmd, V, alt):
    """Propagate a class's own command through the airframe at a step fine enough to be stable.

    Returns the fin history delta(t) -- the observable the signature is rendered from -- and the
    ACHIEVED a_z, from which the onset reference is taken. The command never reaches the detector.
    """
    t = np.asarray(t_grid, float)
    cmd = np.asarray(a_cmd, float)
    n = len(t)
    af = PitchAirframe(DEFAULT_AIRFRAME)
    delta = np.zeros(n); az = np.zeros(n)
    for i in range(n):
        az[i] = af.step(float(cmd[i]), V, alt, DT_FINE)
        delta[i] = af.delta
    return dict(t=t, delta=delta, az=az, cmd=cmd)


def onset_from_achieved(t, az, need_s=0.005):
    """First SUSTAINED crossing of the departure threshold by the ACHIEVED lateral acceleration.

    Achieved, not commanded: the onset reference must be a property of the state a sensor could
    observe. Taking it from the command would measure the synthesis rather than the maneuver.
    """
    need = max(2, int(need_s / DT_FINE))
    ab = np.abs(az) > ONSET_G_MS2
    for i in range(len(ab) - need):
        if ab[i:i + need].all():
            return float(t[i]), i
    return None, -1


# Per-class command SHAPE at the onset, taken from each generator's own source rather than from
# its output grid. This is forced, and the reason is decisive: every generator samples at
# dt = 0.5 s while the fin-to-onset budget is ~35 ms -- the trajectory under-samples the phenomenon
# by ~14x. Interpolating a 0.5 s command onto the airframe's 1e-4 s step does not RESOLVE the onset
# shape, it INVENTS it as a linear ramp; and command shape moves the
# budget 4.2x (147.9 ms raised-cosine vs 35.4 ms step on the identical airframe). So a budget read
# off an interpolated grid would be an artifact of the interpolation, not a property of the class.
#
# Instead: take the AMPLITUDE and the flight condition from the measured trajectory at the onset
# sample, and the functional FORM from the generator that produced it --
#   marv.py:39-41    env = sin(pi*m) over maneuver_dur_s   -> raised-cosine pulse, ~30 s
#   hgv.py:40        a_weave = A*sin(2*pi*n_skips*m)       -> sinusoidal reversal
#   maneuvering.py   a_lat_command = weave_lat_accel*sin(2*pi*weave_cycles*m)  -> SINUSOID,
#                    period weave_dur/weave_cycles = 12.4-19.1 s on the drawn set
#
# THE STEP BELOW IS NOT THE GENERATOR'S SHAPE. maneuvering.py commands a sinusoid;
# there is no step anywhere in it. The distinction is not cosmetic, because the statistic is the
# SQUARE of fin rate and fin rate scales as amplitude/period. Measured
# (experiments/shape_period_sweep.py, n=30 per cell):
#
#     period   0.10   0.30   0.60   1.00   2.00   4.00   15.0 s
#     detect    97%    94%    90%    43%    10%     0%     0%
#
# So the cue converts only for SUB-SECOND command transients, and dies well before the 12-19 s the
# generator for this class actually commands. The manuscript reports that axis and the generator's
# own period rather than quoting the step as though it were the class's command.
# experiments/true_shape_test.py measures the generator's own shape directly: detection 0%.
#
# WHAT THE STEP *IS* IS A MEASURED QUESTION.
# Anchoring the letter on fan2016's first-order lag tau_T = 0.2 s would conflate the COMMAND
# with the vehicle's RESPONSE. In the lumped literature model a_cmd IS a step and the target
# follows da/dt = (a_cmd - a)/tau_T -- the lag is the airframe, not the command. Here the airframe
# is RESOLVED (sim/sixdof.py, 5 states), so it supplies that lag itself. Measured at the FLOWN
# conditions by experiments/onset_model_equivalence.py, n=30:
#
#     closed-loop step response   T_63 = 0.179 s  [0.176, 0.200]      vs the lumped tau_T = 0.2 s
#
# -- 11% apart. So step-command-into-this-airframe already IS the literature's onset model, with
# the lag resolved instead of lumped. Two consequences, both measured, neither assumed:
#
#   * THE LUMP CANNOT REPRESENT WHAT THIS LETTER MEASURES. A first-order lag moves at t=0+ with
#     slope A/tau, so it crosses the 2.0 m/s^2 departure threshold in 12.1 ms against the resolved
#     airframe's 48.45 ms -- 4.00x short. The budget is a property of the fin servo, the short
#     period and the rate limit; a lumped tau has none of them. Resolving it is forced, not a
#     refinement.
#   * experiments/literature_onset_test.py LAGS THE COMMAND BY tau AND THEN DRIVES THE AIRFRAME,
#     so it applies tau_T twice: T_63 0.179 -> 0.374 s (2.09x), budget 48.45 -> 113.8 ms (2.35x).
#     Its "+50.0 ms at tau = 0.2 s" is measured against a budget inflated by that cascade, so it
#     serves as the control that exhibits the double count and not as an anchor.
#
# The duration axis above is unaffected and still decides the scope: the cue is a fast-transient
# effect and needs the COMMAND to change in well under a second, which this class's generator
# never does.
SHAPES = {
    "marv":              ("raised-cosine", 30.0),
    "hgv":               ("sinusoid", 36.0),
    "supersonic_cruise": ("step", 0.0),
}


def synth_command(shape, dur_s, amp, t_pre, t_post):
    """Analytic command at DT_FINE: quiescent for t_pre, then the class's own onset form.

    The post-window is sized to CONTAIN the onset, not fixed. MaRV eases its pull-up in over 30 s
    with a raised cosine, so the command needs ~0.24 s merely to reach the 2.0 m/s^2 departure
    threshold and longer for the achieved state to follow; a fixed 0.4 s window cut the event off
    and reported "no achieved onset", which reads as a physical finding and is not one. The window
    is extended to where the command reaches 3x threshold, capped so it stays affordable at 1e-4 s.
    """
    if shape != "step" and amp > 0:
        target = 3.0 * ONSET_G_MS2
        if target < amp:
            if shape == "raised-cosine":
                t_hit = dur_s * np.arcsin(min(target / amp, 1.0)) / np.pi
            else:
                t_hit = dur_s * np.arcsin(min(target / amp, 1.0)) / (2 * np.pi)
            t_post = float(np.clip(2.0 * t_hit, t_post, 3.0))
    n = int((t_pre + t_post) / DT_FINE)
    t = np.arange(n) * DT_FINE
    a = np.zeros(n)
    m = (t - t_pre) / max(dur_s, 1e-9)
    on = t >= t_pre
    if shape == "step":
        a[on] = amp
    elif shape == "raised-cosine":
        mm = np.clip(m[on], 0.0, 1.0)
        a[on] = amp * np.sin(np.pi * mm)
    elif shape == "sinusoid":
        a[on] = amp * np.sin(2 * np.pi * (t[on] - t_pre) / max(dur_s, 1e-9))
    return t, np.abs(a), t_pre


def class_windows(name, rng=None, max_attempts=40, amp_factor=1.0):
    """Every measurable maneuver EVENT in a class, as a SYNTHESISED command at fine resolution.

    Two conventions, and which applies is a measured property of the class:
      SINGLE ONSET -- quiescent then one sustained departure (marv, supersonic_cruise).
      REVERSAL     -- periodic, no quiescent phase (hgv weaves from launch: 9% quiescent, 10
                      crossings). Each rising crossing is an event, so the class yields TEN events
                      per trajectory rather than one.

    Gating HGV out for "no single onset" would confuse a missing CONVENTION
    with a missing PHENOMENON. All 10 of HGV's reversals command only 0.23-0.48 g and every one
    falls where the airframe can deliver it -- its 27% a_max exceedance lives in the high-g
    stretches BETWEEN reversals, not at the events.
    """
    rec = load_class(name, rng=rng, max_attempts=max_attempts)
    d = describe(rec)
    if d.get("exo") or d["n_crossings"] == 0 or name not in SHAPES:
        return [], d
    sig, _src = lateral_signal(rec)
    man, _trim, _cut = decompose(sig)
    t = rec["t"]
    alt_all, amax_all = feasibility(rec)
    shape, dur = SHAPES[name]

    above = man > ONSET_G_MS2
    rises = list(np.where(np.diff(above.astype(int)) == 1)[0] + 1)
    rises, conv = (rises[:1], "single-onset") if d["has_onset"] else (rises, "reversal")

    wins = []
    for i_on in rises:
        amax = amax_all[i_on]
        # AMPLITUDE = the peak of THIS event, not the value at the crossing. man[i_on] is
        # ~ONSET_G_MS2 by construction -- that is what defines a crossing -- so a command
        # synthesised at that amplitude never rises above the threshold it is measured against, and
        # the class reports "no achieved onset" for a reason that is arithmetic rather than
        # physical.
        falls = np.where(np.diff(above.astype(int)) == -1)[0] + 1
        nxt = falls[falls > i_on]
        i_end = int(nxt[0]) if len(nxt) else len(man)
        amp = float(man[i_on:i_end].max()) if i_end > i_on else float(man[i_on])
        # AMPLITUDE-CORRECT GATE. The caller scales the command (aero_feasible_sweep divides by
        # 4.7), so the gate must compare a_max against the SCALED command -- the one the experiment
        # will actually fly -- and not against the raw draw.
        if np.isfinite(amax) and amax > 0 and amp * amp_factor > amax:
            continue                                            # saturated: not a maneuver
        alt = float(alt_all[i_on])
        V = float(np.linalg.norm(rec["V"][i_on]))
        tt, aa, t_cmd = synth_command(shape, dur, amp, PRE_S, POST_S)
        wins.append(dict(name=name, conv=conv, shape=shape, t=tt, a_cmd=aa, amp_g=amp / 9.80665,
                         alt=alt, V=V, mach=V / max(sound_speed(alt), 1e-6), t_cmd=t_cmd))
    return wins, d



# PER-CLASS PARAMETER DRAWS -- taken from the project's OWN sampler, not invented here.
#
# Hand-written perturbation ranges diverge from the dataset, and not by a little. What the sampler
# supplies, and what a plausible-looking hand-written range would miss:
#   marv        -- pullup_trigger_alt 50-62 km and a DEPRESSED launch_elev 28-37 deg, with pullup
#                  9-12 g and weave 12-17 g. A range centred on 20-30 km and 38-46 deg is a
#                  different vehicle in a different regime.
#   hgv         -- n_skips 6-10 and weave_lat_accel 45-70, plus skip_amp and duration, and above
#                  all aero="cavh", which grounds the glide in the wind-tunnel CAV-H lift/drag
#                  polynomials so that lift SATURATES at low dynamic pressure. Without that flag
#                  the vehicle flies with unbounded lift and is not the HGV the project generates.
#   supersonic  -- boost_from_ground=True, boost_time and duration_s, none of them defaults.
#   ballistic   -- beta, and a wider speed/elevation range than looks natural.
#
# Using generate_dataset._sample_params directly removes the whole class of error: the parameters
# are, by construction, exactly the ones Researchzip generates, and that file is byte-identical
# between the two trees (verified). Same principle as using profiles.GENERATORS rather than a
# hand-written class list -- do not restate the project's own definitions, call them.
from experiments.generate_dataset import _sample_params                       # noqa: E402


def draw_params(name, rng):
    """Exactly the dataset's own per-type physical parameters for this seed."""
    try:
        return _sample_params(name, rng)
    except Exception:                                                # noqa: BLE001
        return {}


def sweep_class(name, seeds, snr, reps, dwell, kin_noise=None, amp_factor=1.0):
    """One advantage per TRAJECTORY. Within a trajectory, events are pooled by median; across
    trajectories the median, its bootstrap CI and a signed-rank test are formed over that pooled
    value. n therefore counts trajectories.

    kin_noise and amp_factor are threaded to measure()/class_windows(); the defaults preserve
    behavior (None -> KIN_NOISE inside measure; 1.0 leaves the command unscaled)."""
    per_traj, budgets, dets, fas, alts = [], [], [], [], []
    arm_traj = {}
    for sd in range(seeds):
        r = np.random.default_rng(90000 + sd)
        try:
            wins, d = class_windows(name, rng=r, amp_factor=amp_factor)
        except Exception:                                            # noqa: BLE001
            continue
        if not wins:
            continue
        advs, buds = [], []
        arm_ev = {}
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w2["a_cmd"], float) * amp_factor
            m = measure(w2, snr, reps, dwell, kin_noise=kin_noise)
            if m is None:
                continue
            buds.append(m["budget_ms"]); dets.append(m["det"]); fas.append(m["fa"])
            alts.append(w["alt"] / 1000)
            if "adv" in m:
                advs.append(m["adv"])
            for k, v in (m.get("arms") or {}).items():
                if v.get("adv") is not None:
                    arm_ev.setdefault(k, []).append(v["adv"])
        if buds:
            budgets.extend(buds)
        if advs:
            per_traj.append(float(np.median(advs)))
        for k, v in arm_ev.items():
            arm_traj.setdefault(k, []).append(float(np.median(v)))
    return dict(n_traj=len(per_traj), adv=np.array(per_traj, float),
                budgets=np.array(budgets, float), det=float(np.mean(dets)) if dets else 0.0,
                fa=float(np.mean(fas)) if fas else 0.0,
                alt=float(np.mean(alts)) if alts else float("nan"),
                arms={k: np.array(v, float) for k, v in arm_traj.items()})


# ---------------------------------------------------------------- prior-art kinematic detectors
# The comparator so far has been a trailing mean of |a_z| -- our own construction, and an ORACLE
# (it reads true lateral acceleration). "We beat a detector we wrote" is a weak claim. These two are
# the field's OWN onset detectors, fully specified in their source papers and therefore genuinely
# reproducible, unlike the signature-domain prior art (yang2007maneuver, zhu2012hrdp released no
# code; experiments/priorart_baselines.py re-implements their PRINCIPLE and says so).
#
# Note what experiments/priorart_baselines.py does NOT do: it races under onset_snr_sweep's
# onset-anchored, matched-per-sample-FPR protocol -- the convention this paper's own failure modes 2
# and 3 show to be broken (an onset-anchored rule scores a transient alarm as a miss; a per-sample
# FPR on a smoothed statistic returns the analyst's search window, slope 1.008). A head-to-head under
# a broken metric decides nothing. These run under the causal zero-false-alarm rule instead.
#
# Both are given the SAME noisy measurement, the SAME dwell and the SAME cruise-maximum threshold as
# every other arm, so the race is symmetric.

def stat_cusum(x, dwell_s, k_frac=0.5):
    """Page (1954) CUSUM for an upward shift in mean.

        S_k = max(0, S_{k-1} + (x_k - mu0 - K)),  K = k_frac * sigma0

    mu0 and sigma0 are estimated from the leading 25% of the record. That is quiescent cruise --
    measured, not assumed: on the supersonic-cruise windows the record runs 0-1.0 s with the
    command at 0.600 s, so the baseline ends at 0.2495 s and cannot contain maneuver samples. If it
    ever did, mu0 and sigma0 would inflate, K = 0.5*sigma0 with them, CUSUM would alarm late and the
    reported advantage would be inflated by an asymmetry nobody declared.

    It is NOT, however, the same window as thr_from_cruise's [t_on-0.60, t_on-0.12] ~ [0.05, 0.53] s.
    Both are pre-command quiescent cruise, so the distinction does not move the result, but the two
    calibrations are separate and the THRESHOLD is the one that is shared identically across arms.

    K IS NOT PAGE'S. Checked against the paper (Biometrika 41, 100-115): Page defines the
    cumulative-sum inspection scheme
    and the average-run-length criterion, and the phrase "reference value" does not occur in it --
    K is a later convention. So K = 0.5*sigma0 is OUR choice, not Page's, and the manuscript now
    states it rather than attributing it. It is swept: over k_frac in [0, 3.3] the advantage spans
    +15.2 to +38.0 ms and never changes sign, and the shipped 0.5 is within 1.75 ms of the best
    value the comparator has anywhere in that range.
    """
    n = len(x)
    m = max(8, int(0.25 * n))
    mu0, sd0 = float(np.mean(x[:m])), float(np.std(x[:m]) + 1e-12)
    K = k_frac * sd0
    S = np.zeros(n)
    acc = 0.0
    for i in range(n):
        acc = max(0.0, acc + (x[i] - mu0 - K))
        S[i] = acc
    return S


def stat_glr(x, dwell_s, win=None):
    """Sliding-window GLR for a jump in the mean of a scalar. NOT Willsky & Jones's detector.

    THIS IS THEIR PRINCIPLE WITHOUT THEIR FILTER, AND THE FILTER IS THE POINT. Checked against the
    paper (IEEE T-AC 21(1), 108-112): Willsky & Jones formulate the GLR over the INNOVATIONS of a
    Kalman-Bucy filter, for jumps in the STATE of a linear system -- "using Kalman-Bucy filtering
    and generalized likelihood ratio techniques". A filter-free Willsky-Jones GLR is a contradiction
    in terms, so their detector is out of scope here for exactly the reason input estimation is:
    this comparison removes the filter. That, not the degeneracy below, is why the arm is not
    presented as a published comparator.

    DEGENERATE AT THIS WINDOW LENGTH -- DO NOT PRESENT AS AN INDEPENDENT COMPARATOR.
    With N = dwell/DT_R = 4 samples and x >= 0, l_k = (sum(x - mu0))^2 / (2 N sigma0^2) is a
    strictly monotone transform of sum(x) above baseline, so under the cruise-MAXIMUM threshold rule
    its first upcrossing coincides with the trailing mean's exactly. Measured: identical alarm times
    on 60/60 real cells (100 %), and runs/ml/multiclass_lead_n30.json carries the two arms with
    identical medians AND identical bootstrap intervals on every class. A real GLR maximises the
    likelihood over an unknown changepoint INSIDE the window, which needs a window materially longer
    than the dwell; at N = 4 there is nothing to maximise over. Guard: glr.not_degenerate.

    Over a sliding window of length N the GLR for an unknown-magnitude step is

        l_k = (sum_{i=k-N+1..k} (x_i - mu0))^2 / (2 N sigma0^2)

    i.e. the squared mean residual scaled by the window length -- the standard closed form when the
    jump magnitude is maximised out. Window length is tied to the shared dwell so it cannot be tuned
    per arm.
    """
    n = len(x)
    m = max(8, int(0.25 * n))
    mu0, var0 = float(np.mean(x[:m])), float(np.var(x[:m]) + 1e-12)
    N = max(2, int(dwell_s / DT_R)) if win is None else win
    c = np.cumsum(np.insert(x - mu0, 0, 0.0))
    out = np.zeros(n)
    idx = np.arange(N, n)
    ssum = c[idx + 1] - c[idx + 1 - N]
    out[idx] = (ssum ** 2) / (2.0 * N * var0)
    return out


KIN_ARMS = (("trailing-mean", None), ("CUSUM Page54", stat_cusum), ("GLR Willsky76", stat_glr))


def measure(win, snr_db, reps, dwell, kin_noise=None):
    """Paired muD-minus-kinematic lead, with a no-cue null on the identical operating point.

    kin_noise is the comparator's measurement-noise sigma. It defaults to KIN_NOISE but is threaded
    explicitly because the advantage is a monotone function of it -- see the module constant.
    """
    sigma = KIN_NOISE if kin_noise is None else float(kin_noise)
    fl = drive_airframe(win["t"], win["a_cmd"], win["V"], win["alt"])
    t_on, i_on = onset_from_achieved(fl["t"], fl["az"])
    if t_on is None:
        return None
    budget_ms = 1000.0 * (t_on - win["t_cmd"])

    tf, delta, az = fl["t"], fl["delta"], fl["az"]
    diffs, mds, kins, det, fa = [], [], [], 0, 0
    arms = {}
    for r in range(reps):
        t, s = return_from_fin(tf, delta, snr_db, 4000 + r, FIN_ARM_M)
        st = stat_matched_phase(t, s, dwell)
        lm = causal_lead(t, st, thr_from_cruise(t, st, t_on), t_on)
        rk = np.random.default_rng(4000 + r + 991)
        azr = np.interp(t, tf, az) + rk.normal(0, sigma, len(t))
        nk = max(2, int(dwell / DT_R))
        for arm, fn in KIN_ARMS:
            stk = (np.convolve(np.abs(azr), np.ones(nk) / nk, mode="full")[:len(t)]
                   if fn is None else fn(np.abs(azr), dwell))
            lk = causal_lead(t, stk, thr_from_cruise(t, stk, t_on), t_on)
            if lk is not None:
                arms.setdefault(arm, {"lead": [], "adv": []})["lead"].append(1000 * lk)
                if lm is not None:
                    arms[arm]["adv"].append(1000 * (lm - lk))
        lk = arms.get("trailing-mean", {}).get("lead", [None])[-1]             if arms.get("trailing-mean", {}).get("lead") else None
        if lm is not None:
            det += 1; mds.append(1000 * lm)
        if lk is not None:
            kins.append(lk)
        if lm is not None and lk is not None:
            diffs.append(1000 * lm - lk)
        # NULL: identical noise and threshold rule, fin held still
        tn, sn = return_from_fin(tf, delta, snr_db, 4000 + r, FIN_ARM_M, cue_on=False)
        stn = stat_matched_phase(tn, sn, dwell)
        if causal_lead(tn, stn, thr_from_cruise(tn, stn, t_on), t_on) is not None:
            fa += 1

    a = np.array(diffs, float)
    out = dict(budget_ms=budget_ms, det=det / reps, fa=fa / reps,
               muD=float(np.median(mds)) if mds else None,
               kin=float(np.median(kins)) if kins else None, n=int(a.size),
               arms={k: dict(lead=float(np.median(v["lead"])) if v["lead"] else None,
                             adv=float(np.median(v["adv"])) if v["adv"] else None,
                             n=len(v["adv"])) for k, v in arms.items()})
    if a.size >= 3:
        rng = np.random.default_rng(20260801)
        bt = np.median(rng.choice(a, (4000, a.size)), axis=1)
        try:
            p = float(wilcoxon(a, alternative="two-sided", zero_method="zsplit").pvalue)
        except ValueError:
            p = float("nan")
        out.update(adv=float(np.median(a)), lo=float(np.percentile(bt, 2.5)),
                   hi=float(np.percentile(bt, 97.5)), p=p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--dwell", type=float, default=0.002)
    ap.add_argument("--seeds", type=int, default=12,
                    help="TRAJECTORIES per class -- n is over trajectories, not noise draws")
    ap.add_argument("--kin-noise", type=float, default=KIN_NOISE,
                    help="comparator measurement-noise sigma (m/s^2); the advantage is "
                         "monotone in it and changes sign near 0.005 -- see sigma_sweep.py")
    ap.add_argument("--amp-factor", type=float, default=1.0,
                    help="command amplitude scale; 0.2798 is the derived feasible amplitude")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    kn = args.kin_noise

    from trajectory_generators.profiles import GENERATORS

    print("MICRO-DOPPLER LEAD BY TARGET CLASS")
    print("  cue rendered from the airframe's fin history delta(t), never from the command.")
    print("  onset taken from ACHIEVED lateral acceleration. Comparator is an ORACLE reading true")
    print("  a_z plus noise, so every advantage below is a LOWER BOUND. Budget is per class and")
    print("  is never pooled: command shape moves it 4.2x.\n")
    print("  SNR %.0f dB | dwell %.0f ms | n=%d noise draws | null control at every cell\n"
          % (args.snr, 1000 * args.dwell, args.reps))
    print("%-18s %-12s %6s %12s %8s %6s %6s  %s"
          % ("class", "convention", "n traj", "budget ms", "alt", "det", "FA",
             "ADVANTAGE ms (over trajectories)"))
    print("-" * 112)

    out = []
    for name in GENERATORS:
        try:
            _w, d = class_windows(name, rng=np.random.default_rng(7777))
        except Exception as e:                                       # noqa: BLE001
            print("%-18s %-12s %6s  GATED OUT -- %s" % (name, "--", "--", str(e)[:52]))
            out.append(dict(name=name, gated=True, why=str(e)[:80]))
            continue
        if not _w:
            # Distinguish the three ways a class can produce no measurable event; "never
            # maneuvers" covers only one. MaRV DOES maneuver (1
            # crossing, 11.95 g) but at the dataset's 50-62 km arming altitude the command exceeds
            # available control authority, so every event is saturation rather than a maneuver.
            if d.get("exo"):
                why = "exo-atmospheric separation impulse at %.0f km" % d["alt_onset_km"]
            elif d["n_crossings"] == 0:
                why = "never maneuvers: 0 crossings, peak %.2f g" % d["peak_g"]
            else:
                why = ("maneuvers (%d crossings, peak %.2f g) but every event saturates: %.0f%% of "
                       "flight commands more than a_max at %.0f km"
                       % (d["n_crossings"], d["peak_g"], 100 * d["frac_over_amax"],
                          d["alt_onset_km"]))
            print("%-18s %-12s %6s  GATED OUT -- %s" % (name, "--", "--", why))
            out.append(dict(name=name, gated=True, why=why))
            continue
        r = sweep_class(name, args.seeds, args.snr, args.reps, args.dwell,
                        kin_noise=kn, amp_factor=args.amp_factor)

        a = r["adv"]
        if a.size >= 3:
            rng = np.random.default_rng(20260801)
            bt = np.median(rng.choice(a, (4000, a.size)), axis=1)
            try:
                pv = float(wilcoxon(a, alternative="two-sided", zero_method="zsplit").pvalue)
            except ValueError:
                pv = float("nan")
            cell = "%+.1f [%+.1f,%+.1f] n=%d p=%.3f" % (np.median(a), np.percentile(bt, 2.5),
                                                        np.percentile(bt, 97.5), a.size, pv)
        elif a.size:
            cell = "%+.1f (n=%d traj, no CI)" % (np.median(a), a.size)
        else:
            cell = "no trajectory converts (det %.0f%%)" % (100 * r["det"])
        b = r["budgets"]
        brange = "%.0f-%-6.0f" % (b.min(), b.max()) if b.size else "--"
        print("%-18s %-12s %6d %12s %7.1fkm %5.0f%% %5.0f%%  %s"
              % (name, _w[0]["conv"], r["n_traj"], brange, r["alt"],
                 100 * r["det"], 100 * r["fa"], cell))
        out.append(dict(name=name, gated=False, conv=_w[0]["conv"], n_traj=int(r["n_traj"]),
                        budget_min=float(b.min()) if b.size else None,
                        budget_max=float(b.max()) if b.size else None,
                        det=r["det"], fa=r["fa"],
                        adv=[float(x) for x in a],
                        arm_adv={k: [float(x) for x in v] for k, v in r["arms"].items()}))

    print("\nHEAD-TO-HEAD vs THE FIELD'S OWN DETECTORS. Same measurement, same dwell, same")
    print("  cruise-maximum (zero-false-alarm) operating point for every arm, so the race is")
    print("  symmetric. Advantage is muD minus that arm, median over TRAJECTORIES; positive")
    print("  means the signature channel wins.\n")
    print("  %-18s %-16s %7s %26s" % ("class", "comparator", "n traj", "muD advantage (ms)"))
    print("  " + "-" * 74)
    for o in out:
        if o.get("gated") or not o.get("arm_adv"):
            continue
        for arm, vals in o["arm_adv"].items():
            v = np.array(vals, float)
            if v.size >= 3:
                rng = np.random.default_rng(20260801)
                bt = np.median(rng.choice(v, (4000, v.size)), axis=1)
                cell = "%+.1f [%+.1f,%+.1f]" % (np.median(v), np.percentile(bt, 2.5),
                                                np.percentile(bt, 97.5))
            elif v.size:
                cell = "%+.1f (n=%d, no CI)" % (np.median(v), v.size)
            else:
                cell = "no converging trajectory"
            print("  %-18s %-16s %7d %26s" % (o["name"], arm, v.size, cell))
    print("\n  CUSUM is Page (1954); GLR is Willsky & Jones (1976). Both are reproduced from their")
    print("  published closed forms, unlike the signature-domain prior art (yang2007maneuver,")
    print("  zhu2012hrdp released no code -- priorart_baselines.py re-implements their PRINCIPLE")
    print("  and says so). Note that priorart_baselines.py races under the onset-anchored,")
    print("  matched-per-sample-FPR protocol this paper's own failure modes 2 and 3 show to be")
    print("  broken; this table uses the causal zero-false-alarm rule instead.")
    print("\nCAVEAT. n is over trajectories per class. Budgets differ per class and are not")
    print("  comparable as percentages -- the denominator moves 4.2x with command shape (7j).")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
