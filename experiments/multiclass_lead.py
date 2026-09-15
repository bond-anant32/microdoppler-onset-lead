"""experiments/multiclass_lead.py -- micro-Doppler onset lead per target class.

Each class's command is propagated through sim.sixdof.PitchAirframe at that class's own flight
condition. experiments/class_profiles.py measures, per class, whether a maneuver onset exists and
what fraction of the flight commands more than a_max_at(alt, mach). At HGV's 40 km glide 91 % of
the flight commands more lateral acceleration than the airframe can deliver, and achieved a_z
saturates near 1.03 g against an 8.0 g command with the fin on its rail. MaRV is numerically
unstable at the generator's dt = 0.5 s, so the airframe is integrated at DT_FINE.

The five classes:

    supersonic_cruise  events form and convert; the class reported in the paper
    marv               maneuvers (11.95 g), but at its ~52 km arming altitude the amplitude-scaled
                       a_max check rejects every event as saturation
    hgv                events form (8 per trajectory, reversal convention); the weave has a 146 s
                       median period, and over 4 statistic families x 6 dwells from 2 ms to 5 s,
                       0 of 15 formable cells convert (experiments/hgv_detector_sweep.py).
                       High-detection cells match their own no-cue nulls (100 % vs 100 % at the
                       5 s dwell).
    ballistic          never maneuvers: 0 crossings, peak 0.10 g; lead undefined
    mirv               0 crossings, peak 0.04 g, exo-atmospheric dispense impulse; lead undefined

Event counts per class:

    python -c "import numpy as np; from experiments.multiclass_lead import class_windows; \
        print([(n, len(class_windows(n, rng=np.random.default_rng(7777))[0])) \
               for n in ('supersonic_cruise','marv','hgv','ballistic','mirv')])"

The signature is rendered from the airframe's fin deflection delta(t), which is downstream of the
physics and observable in principle. The onset budget depends on command shape (48.45 ms for a
step and 712.55 ms for a 30 s raised cosine on the supersonic-cruise draws,
experiments/onset_model_equivalence.py), so budgets are computed per class and not pooled, and
leads are reported in absolute ms alongside each class's budget.

The kinematic comparator is a trailing mean of |a_z| formed from the true lateral acceleration plus
additive noise. It thresholds the true state without estimating it, so no tracking-filter lag
enters the comparison.

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

DT_FINE = 1e-4               # airframe integration step (s); MaRV is unstable at the generator's 0.5 s
PRE_S, POST_S = 0.60, 0.40   # window around the command onset: cruise for the threshold, then event
FIN_ARM_M = 0.30
KIN_NOISE = 0.3              # default comparator measurement-noise sigma (m/s^2). The muD lead is
                             # constant in sigma, so the advantage (muD lead - kin lead(sigma)) is
                             # monotone in it and changes sign between sigma = 0 and 0.003 m/s^2.
                             # See experiments/sigma_sweep.py.


def drive_airframe(t_grid, a_cmd, V, alt):
    """Propagate a command through PitchAirframe at DT_FINE.

    Returns dict(t, delta, az, cmd): the fin history delta(t), from which the signature is
    rendered, and the achieved a_z, from which the onset reference is taken.
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
    """First crossing of ONSET_G_MS2 by the achieved lateral acceleration sustained for need_s.

    The onset reference is a property of the achieved state, which a sensor could observe.
    Returns (t_on, index), or (None, -1) if no sustained crossing occurs.
    """
    need = max(2, int(need_s / DT_FINE))
    ab = np.abs(az) > ONSET_G_MS2
    for i in range(len(ab) - need):
        if ab[i:i + need].all():
            return float(t[i]), i
    return None, -1


# Per-class command shape at the onset. The generators sample at dt = 0.5 s, about 10x coarser than
# the ~48 ms fin-to-onset budget, and interpolating that grid onto the 1e-4 s airframe step turns
# every onset into a linear ramp. The budget depends strongly on shape (48.45 ms for a step,
# 712.55 ms for a 30 s raised cosine; experiments/onset_model_equivalence.py), so the command is
# synthesised analytically: amplitude and flight condition from the trajectory at the onset sample,
# functional form from the generator source.
#   marv.py:39-41    env = sin(pi*m) over maneuver_dur_s   -> raised-cosine pulse, ~30 s
#   hgv.py:40        a_weave = A*sin(2*pi*n_skips*m)       -> sinusoidal reversal
#   maneuvering.py   a_lat_command = weave_lat_accel*sin(2*pi*weave_cycles*m)  -> sinusoid,
#                    period weave_dur/weave_cycles = 12.4-19.1 s on the drawn set
#
# supersonic_cruise uses a step. Its generator, maneuvering.py, commands a sinusoid. The statistic
# is the square of fin rate, and fin rate scales as amplitude/period, so detection falls with
# command period (experiments/shape_period_sweep.py, n=30 per cell):
#
#     period   0.10   0.30   0.60   1.00   2.00   4.00   15.0 s
#     detect    97%    94%    90%    43%    10%     0%     0%
#
# The cue converts for sub-second command transients. At the generator's own 12-19 s period
# detection is 0% (experiments/true_shape_test.py).
#
# A step into the resolved airframe reproduces the literature's onset model. In the lumped model
# (fan2016) a_cmd is a step and the target follows da/dt = (a_cmd - a)/tau_T with tau_T = 0.2 s.
# Here the 5-state airframe (sim/sixdof.py) supplies that lag. At the flown conditions its
# closed-loop step response has T_63 = 0.179 s, range 0.176-0.200 s over n=30 draws
# (experiments/onset_model_equivalence.py), 11% from the lumped 0.2 s.
#   * A first-order lag moves at t=0+ with slope A/tau and crosses the 2.0 m/s^2 departure
#     threshold in 12.1 ms; the resolved airframe takes 48.45 ms, 4.00x longer. The budget is set
#     by the fin servo, the short period and the rate limit, none of which a lumped tau contains.
#   * Lagging the command by tau before driving the airframe applies tau_T twice:
#     T_63 0.179 -> 0.374 s (2.09x), budget 48.45 -> 113.8 ms (2.35x).
SHAPES = {
    "marv":              ("raised-cosine", 30.0),
    "hgv":               ("sinusoid", 36.0),
    "supersonic_cruise": ("step", 0.0),
}


def synth_command(shape, dur_s, amp, t_pre, t_post):
    """Analytic command at DT_FINE: zero for t_pre, then the class's onset form at amplitude amp.

    shape is "step", "raised-cosine" (duration dur_s) or "sinusoid" (period dur_s). For non-step
    shapes the post-window is extended to twice the time the command takes to reach 3x the
    departure threshold, bounded below by t_post and above by 3.0 s. MaRV's 30 s raised cosine needs
    ~0.24 s to reach the 2.0 m/s^2 threshold, and the achieved state follows later still.

    Returns (t, |a_cmd|, t_pre).
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
    """Every measurable maneuver event in a class, as a synthesised command at DT_FINE.

    The event convention is a measured property of the class:
      single onset  quiescent, then one sustained departure (marv, supersonic_cruise)
      reversal      periodic with no quiescent phase (hgv weaves from launch: 9% quiescent,
                    10 crossings); each rising crossing is an event, giving ten per trajectory

    HGV's 10 reversals command 0.23-0.48 g, all within the airframe's authority; its 27% a_max
    exceedance lies in the high-g stretches between reversals.

    Events whose scaled amplitude (amp * amp_factor) exceeds a_max are skipped as saturated.
    Returns (list of window dicts, describe() summary of the class).
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
        # Amplitude is the peak of this event. man[i_on] is ~ONSET_G_MS2 by construction, so a
        # command synthesised at the crossing value would never rise above the threshold.
        falls = np.where(np.diff(above.astype(int)) == -1)[0] + 1
        nxt = falls[falls > i_on]
        i_end = int(nxt[0]) if len(nxt) else len(man)
        amp = float(man[i_on:i_end].max()) if i_end > i_on else float(man[i_on])
        # Callers may scale the command (amp_factor), so a_max is compared against the scaled
        # amplitude that is actually flown.
        if np.isfinite(amax) and amax > 0 and amp * amp_factor > amax:
            continue                                            # saturated event
        alt = float(alt_all[i_on])
        V = float(np.linalg.norm(rec["V"][i_on]))
        tt, aa, t_cmd = synth_command(shape, dur, amp, PRE_S, POST_S)
        wins.append(dict(name=name, conv=conv, shape=shape, t=tt, a_cmd=aa, amp_g=amp / 9.80665,
                         alt=alt, V=V, mach=V / max(sound_speed(alt), 1e-6), t_cmd=t_cmd))
    return wins, d



# Per-class parameter draws come from the dataset sampler, generate_dataset._sample_params, so they
# match the generated dataset exactly. The sampler's ranges include:
#   marv        pullup_trigger_alt 50-62 km, depressed launch_elev 28-37 deg, pullup 9-12 g,
#               weave 12-17 g
#   hgv         n_skips 6-10, weave_lat_accel 45-70, skip_amp, duration, and aero="cavh", which
#               uses the CAV-H wind-tunnel lift/drag polynomials so lift saturates at low dynamic
#               pressure (without it the vehicle flies with unbounded lift)
#   supersonic  boost_from_ground=True, boost_time and duration_s, none of them defaults
#   ballistic   beta, and a wide speed/elevation range
from experiments.generate_dataset import _sample_params                       # noqa: E402


def draw_params(name, rng):
    """The dataset's per-type physical parameters for this rng, or {} if the sampler raises."""
    try:
        return _sample_params(name, rng)
    except Exception:                                                # noqa: BLE001
        return {}


def sweep_class(name, seeds, snr, reps, dwell, kin_noise=None, amp_factor=1.0):
    """Advantage per trajectory for one class, over `seeds` trajectories.

    Within a trajectory, events are pooled by median; main() forms the median, bootstrap CI and
    signed-rank test across trajectories, so n counts trajectories. kin_noise is passed to
    measure() (None uses KIN_NOISE); amp_factor scales the command and is passed to
    class_windows() (1.0 leaves it unscaled)."""
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
# Two published kinematic change detectors, alongside the trailing mean of |a_z|. Every arm receives
# the same noisy measurement, the same dwell and the same cruise-maximum threshold, and is scored
# under the causal zero-false-alarm rule.

def stat_cusum(x, dwell_s, k_frac=0.5):
    """Page (1954) CUSUM for an upward shift in mean.

        S_k = max(0, S_{k-1} + (x_k - mu0 - K)),  K = k_frac * sigma0

    mu0 and sigma0 are estimated from the leading 25% of the record. On the supersonic-cruise
    windows the record runs 0-1.0 s with the command at 0.600 s, so the baseline ends at 0.2495 s,
    inside quiescent cruise. The detection threshold comes separately from thr_from_cruise over
    [t_on-0.60, t_on-0.12] ~ [0.05, 0.53] s, also pre-command cruise, and is shared by all arms.

    Page (Biometrika 41, 100-115) defines the cumulative-sum inspection scheme and the
    average-run-length criterion; the reference value K is a later convention. Over k_frac in
    [0, 3.3] the advantage spans +15.2 to +38.0 ms with no sign change, and the default 0.5 is
    within 1.75 ms of the comparator's best value in that range.
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
    """Sliding-window GLR for a jump in the mean of a scalar, applied without a filter.

    Over a sliding window of length N the GLR for an unknown-magnitude step is

        l_k = (sum_{i=k-N+1..k} (x_i - mu0))^2 / (2 N sigma0^2)

    the squared mean residual scaled by the window length, with the jump magnitude maximised out.
    N = dwell_s/DT_R unless win is given, so the window follows the shared dwell.

    Willsky and Jones (IEEE T-AC 21(1), 108-112) formulate the GLR over the innovations of a
    Kalman-Bucy filter for jumps in the state of a linear system. This arm applies the
    likelihood-ratio statistic directly to the measurement, and the comparison in this script has
    no filter.

    At the default 2 ms dwell N = 4 samples. With x >= 0, l_k is then a strictly monotone transform
    of sum(x) above baseline, so under the cruise-maximum threshold its first upcrossing coincides
    with the trailing mean's: alarm times are identical on 60/60 cells, and
    runs/ml/multiclass_lead_n30.json shows identical medians and bootstrap intervals for the two
    arms on every class. Maximising over an unknown changepoint inside the window requires a window
    materially longer than the dwell.
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
    """Paired muD-minus-kinematic lead, with a no-cue null at the identical operating point.

    kin_noise is the comparator's measurement-noise sigma (None uses KIN_NOISE); the advantage is
    monotone in it. Returns None if the achieved state has no onset. Otherwise returns a dict with
    budget_ms, det, fa, muD and kin leads (ms), n, and per-arm results under "arms"; adv, lo, hi
    (bootstrap 95% interval) and p (Wilcoxon signed-rank) are added when n >= 3.
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
        # Null: identical noise and threshold rule, fin held still
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
                    help="trajectories per class; reported n counts trajectories")
    ap.add_argument("--kin-noise", type=float, default=KIN_NOISE,
                    help="comparator measurement-noise sigma (m/s^2); the advantage is "
                         "monotone in it and changes sign between 0 and 0.003 (sigma_sweep.py)")
    ap.add_argument("--amp-factor", type=float, default=1.0,
                    help="command amplitude scale; 0.2798 is the derived feasible amplitude")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    kn = args.kin_noise

    from trajectory_generators.profiles import GENERATORS

    print("Micro-Doppler lead by target class")
    print("  Cue rendered from the airframe's fin history delta(t).")
    print("  Onset taken from the achieved lateral acceleration. The comparator reads the true")
    print("  a_z plus noise, with no tracking filter. Budgets are computed per")
    print("  class and not pooled, because the budget depends strongly on command shape.\n")
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
            print("%-18s %-12s %6s  excluded -- %s" % (name, "--", "--", str(e)[:52]))
            out.append(dict(name=name, gated=True, why=str(e)[:80]))
            continue
        if not _w:
            # A class yields no measurable event in one of three ways: exo-atmospheric separation,
            # no threshold crossing, or every event saturating. MaRV crosses once (11.95 g), but at
            # the dataset's 50-62 km arming altitude the command exceeds the available control
            # authority.
            if d.get("exo"):
                why = "exo-atmospheric separation impulse at %.0f km" % d["alt_onset_km"]
            elif d["n_crossings"] == 0:
                why = "never maneuvers: 0 crossings, peak %.2f g" % d["peak_g"]
            else:
                why = ("maneuvers (%d crossings, peak %.2f g) but every event saturates: %.0f%% of "
                       "flight commands more than a_max at %.0f km"
                       % (d["n_crossings"], d["peak_g"], 100 * d["frac_over_amax"],
                          d["alt_onset_km"]))
            print("%-18s %-12s %6s  excluded -- %s" % (name, "--", "--", why))
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

    print("\nComparison with kinematic change detectors. Every arm uses the same measurement,")
    print("  the same dwell and the same cruise-maximum (zero-false-alarm) threshold.")
    print("  Advantage is the muD lead minus that arm's lead, median over trajectories;")
    print("  positive means the signature channel alarms first.\n")
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
    print("\n  CUSUM is the Page (1954) cumulative-sum scheme with reference value K = 0.5 sigma0,")
    print("  a later convention. The GLR arm is a sliding-window GLR for a jump in the mean,")
    print("  applied to the measurement without the Kalman-Bucy filter of Willsky and Jones")
    print("  (1976). At the default 2 ms dwell (4 samples) its first upcrossing coincides")
    print("  with the trailing mean's. All arms are scored under the causal zero-false-alarm")
    print("  rule.")
    print("\nn counts trajectories per class. Budgets differ between classes and are not")
    print("  comparable as percentages, since the budget depends strongly on command shape.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
