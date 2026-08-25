"""experiments/onset_snr_sweep.py -- anticipation-recovery experiment (paper #1), rigorous version.

Model-conditional thesis, not a strict bound on this script's outputs: in the
lagged-autopilot model, the commanded body-pitch/control response leads the velocity-vector turn by
tau_T, and micro-Doppler reads the command while a bulk-kinematic detector reads the response.
IMPORTANT METRIC NOTE: leads here are measured against the ONSET_G *threshold crossing*, which lags
the physical response by the accel ramp -- so raw muD leads EXCEED tau (= tau + ramp; the tau bound
holds only vs the response onset t_c+tau). The kinematic detector can also show small positive lead
vs the threshold crossing at loose FPR. The paper's causal claim is therefore NEITHER the raw lead
nor the raw advantage but the PAIRED PER-TRACK DELTA vs a cue-severed control (paired_delta.py,
draft 5.1c). Synthesis convention: each spectrogram is rendered from the INSTANTANEOUS state (zero
dwell-fill latency) -- absolute leads are upper bounds under that convention (draft 5.2).

RIGOR (fixes the earlier >tau artifact):
  * t_onset is FIXED = first sustained crossing of a small physical lateral-accel threshold (the real
    departure from cruise). BOTH detectors' leads are measured against THIS single reference -- never
    against each other -- so a late kinematic alarm can't inflate the muD lead past tau.
  * ONSET-ANCHORED sustained alarm (walk back through the contiguous above-threshold run leading INTO
    the onset, tolerating <=1 dropout). A distant spurious cruise false alarm cannot register as lead,
    and the lead is bounded by when the cue physically rises (t_onset - tau_eff) -> lead_muD <= tau_eff.
  * muD statistic MATCHED to the cue: the 3-10 Hz band power of the spectrogram's per-column energy
    (the leading pitch/flutter modulation is sub-Doppler-bin, so it is a slow TIME-axis oscillation,
    not a Doppler spread). Kinematic statistic = |lateral accel| (SNR-independent).
  * 30 tracks/class, IQR reported.

HONESTY: signatures are physics-based surrogates; tau (ACTUATOR_LAG_S=1.2 s) + the cue model are
calibrated, not measured. This quantifies how much of the causal window survives realistic muD SNR.

    python experiments/onset_snr_sweep.py --tracks 30 --out runs/ml/snr_sweep.npz
"""
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajectory_generators.corridors import random_long_corridor          # noqa: E402
from trajectory_generators.dynamics import altitude_of                    # noqa: E402
from trajectory_generators.maneuvering import ACTUATOR_LAG_S              # noqa: E402
from experiments.generate_dataset import _build_objects                   # noqa: E402
from sim.signatures import micro_doppler, control_series_from_trajectory, Control  # noqa: E402

H = W = 64
_PRF = 2000.0; _HOP = (2 * H) // 2                # micro_doppler slow-time params
_COL_DT = _HOP / _PRF                             # 0.032 s between spectrogram columns
_CUE_LO, _CUE_HI = 3.0, 10.0                      # cue band (4-8 Hz pitch/flutter, sub-Doppler-bin)
ONSET_G = 2.0                                     # m/s^2 (~0.2 g): the physical kinematic-departure threshold
# Cue-filter width handed to control_series_from_trajectory. Default 1.0 reproduces every prior run
# byte-for-byte. NOTE what that default actually is on the paper's tracks: k = max(3, int(1.0/dt))
# with dt = 0.5 s gives k = 3 samples = a 1.5 s causal boxcar -- the max(3,...) floor binds, so the
# filter is HALF AGAIN wider than the "1.0 s" the parameter name implies. That suppresses the very
# lead this experiment measures (0.15-0.45 s under first-order lag), and a real fin deflects in
# ~12 ms. Sweep it with --smooth / CUE_SMOOTH_S to separate the physical effect from our own filter.
CUE_SMOOTH_S = float(os.environ.get("CUE_SMOOTH_S", "1.0"))

# How a trajectory-grid series is resampled onto the 0.1 s detection grid.
#   'causal' -- previous-sample hold: the value at t is the last trajectory sample at or before t.
#   'linear' -- np.interp, the behaviour that produced the pre-registered cells in runs/ml/tau_floor.
# WHY THIS IS NOT COSMETIC. The cue is a step: control_series_from_trajectory
# normalises by ref = max(percentile(src,95), 1.0), and since the weave occupies ~3% of the record
# the 95th percentile of |cmd_lat_accel| is 0, so ref falls back to 1.0 against a commanded peak of
# 92-138 m/s^2 and lead_cue saturates to 1 within one sample. np.interp then spreads that step
# BACKWARDS across one trajectory sample (0.5 s). Measured at ACTUATOR_LAG_S=0, where t_onset ==
# t_cmd on all 30 tracks and no maneuver information exists before t_cmd at all, the muD alarm
# landed a median 0.55 s BEFORE the command, on 30 of 30 tracks. Switching this to 'causal' removes
# exactly 0.50 s -- one trajectory sample -- on 29 of 30, at every tau and in both command models.
# It is a fixed offset, so the EXCESS column of Table I is unchanged; what it moves is the tau=0
# floor and every absolute lead reported in the paper.
RESAMPLE = os.environ.get("RESAMPLE", "causal").lower()

# Detection-grid extent, relative to t_onset, shared by EVERY arm (muD here, IMM and oracle via
# kinematic_baselines.lead_of). Must not depend on tau: see the note in measure(). EVAL_PRE abuts the
# cruise calibration window, which ends at t_onset - 6.0, so evaluation and calibration never overlap.
EVAL_PRE, EVAL_POST = 6.0, 3.0


def resample(fg, tt, y, how=None):
    """Trajectory series -> detection grid. See RESAMPLE above; 'causal' never reads the future."""
    if (how or RESAMPLE) == "linear":
        return np.interp(fg, tt, y)
    idx = np.searchsorted(np.asarray(tt), np.asarray(fg), side="right") - 1
    return np.asarray(y)[np.clip(idx, 0, len(y) - 1)]


def cue_band_power(spec, lo=_CUE_LO, hi=_CUE_HI):
    """muD statistic: band power of the per-column energy in [lo, hi] Hz, normalized by mean energy
    (robust to the per-spectrogram max-norm). Default band 3-10 Hz. The band is NOT
    hand-granted -- experiments/band_select.py selects it on DISJOINT calibration tracks and the
    3-10 Hz choice must be reproduced there before it is used in any headline table."""
    ce = spec.sum(axis=0).astype(float); mu = ce.mean()
    if mu < 1e-9 or len(ce) < 8:
        return 0.0
    P = np.abs(np.fft.rfft(ce - mu)) ** 2
    f = np.fft.rfftfreq(len(ce), d=_COL_DT)
    return float(P[(f >= lo) & (f <= hi)].sum() / (mu * mu * len(ce)))


def loo_leads(cal, dec, fpr=None, rel=None):
    """LEAVE-ONE-TRACK-OUT threshold, applied to a whole cell's saved calibration/decision traces.

    THE LOCKED CONVENTION -- this is the threshold behind Table I of the paper, and the reason it is
    not the per-track quantile of eq. (6)'s earlier form: in that form the evaluated track supplies
    its own threshold, so every track is scored at an operating point fitted to itself. An external
    review flagged it as calibration leakage and it is: a deployable detector carries a threshold set
    off-line, not one refitted to the record it is about to decide on.

    Here the threshold for track i is the (1-fpr) quantile of the cruise samples POOLED OVER ALL
    OTHER tracks of the same cell. It is held out by construction, and -- the part that matters for
    this paper -- it is applied to BOTH arms by the same rule. Calibrating one arm and not the other
    reintroduces exactly the arm asymmetry the study is about, so there is deliberately no per-arm
    switch anywhere in this code path.

      cal  (n, Kc)  per-track cruise calibration samples, NaN-padded to a common width
      dec  (n, Kd)  per-track decision samples on the fixed 0.1 s grid
      rel           the grid `dec` is on, relative to t_onset (default: the shared EVAL window)

    Returns an (n,) array of leads, NaN where the track raised no scoreable alarm. Nothing is
    re-rendered and no RNG is touched, so the self-calibrated and held-out numbers differ by the
    calibration change alone.
    """
    if fpr is None:
        fpr = 0.10
    if rel is None:
        rel = np.arange(-EVAL_PRE, EVAL_POST, 0.1)
    cal = np.asarray(cal, float)
    dec = np.asarray(dec, float)
    n = cal.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        pool = np.concatenate([cal[j][np.isfinite(cal[j])] for j in range(n) if j != i])
        d = dec[i][:len(rel)]
        if pool.size < 5 or not np.isfinite(d).all():
            continue
        lead = onset_anchored_lead(rel, d, float(np.quantile(pool, 1 - fpr)), 0.0)
        if lead is not None:
            out[i] = lead
    return out


def onset_anchored_lead(t, stat, thr, t_onset, need=2, max_gap=1, late_tol=2.0):
    """Lead of the ONSET-ANCHORED sustained alarm: the contiguous run of above-threshold samples leading
    INTO t_onset, walked back tolerating <=max_gap dropouts. lead = t_onset - t_run_start (>0 = early).
    If not alarming at onset, allow a late detection within late_tol (negative lead); None = missed."""
    above = stat > thr
    n = len(t)
    j = int(np.searchsorted(t, t_onset, side="right")) - 1
    j = min(max(j, 0), n - 1)
    # `need` is the anti-blip sustain requirement, enforced on BOTH paths so that a single-sample
    # threshold blip cannot count as an alarm: the run must contain >= need above-threshold samples.
    if not above[j]:
        fwd = [k for k in range(j, n) if t[k] <= t_onset + late_tol and above[k]
               and sum(above[k:min(k + need, n)]) >= min(need, n - k)]
        return (t_onset - t[fwd[0]]) if fwd else None
    start, k, gaps, run_len = j, j, 0, 1
    while k - 1 >= 0:
        if above[k - 1]:
            start = k - 1; k -= 1; run_len += 1
        elif gaps < max_gap and k - 2 >= 0 and above[k - 2]:
            gaps += 1; start = k - 2; k -= 2; run_len += 1
        else:
            break
    # A run that BEGINS at onset and continues forward is sustained too, so count the
    # forward continuation before demoting, else a {j, j+1} run (exactly `need` samples spanning
    # onset) is demoted and then missed by the late path, dropping the pair.
    kf = j
    while run_len < need and kf + 1 < n and above[kf + 1] and t[kf + 1] <= t_onset + late_tol:
        run_len += 1; kf += 1
    if run_len < need:                                   # onset-anchored run is a blip -> late path
        fwd = [k2 for k2 in range(j + 1, n) if t[k2] <= t_onset + late_tol and above[k2]
               and sum(above[k2:min(k2 + need, n)]) >= min(need, n - k2)]
        return (t_onset - t[fwd[0]]) if fwd else None
    return t_onset - t[start]


def build_track(cls, seed, onset_g=ONSET_G, cue_mode="cmd"):
    """cue_mode:
      'cmd' -- the signature cue is the COMMAND channel (the paper's model; leads by tau).
      'kin' -- the signature cue is the causal-smoothed KINEMATIC envelope (no command knowledge):
               the muD machinery still sees a maneuver modulation, but one that cannot lead.
               EXPECT advantage <= 0. If it stays positive, the pipeline manufactures lead -> tautology.
      'off' -- the signature cue is ZERO (kinematics untouched): the muD statistic sees no
               onset-related modulation at all. EXPECT detection to collapse to the FPR floor.
    Onset references (t_cmd, t_onset) are ALWAYS computed from the command channel so all three
    modes are measured against the identical physical reference."""
    rng = np.random.default_rng(seed)
    mn, mx = (1400.0, 2200.0) if cls == "marv" else (2200.0, 3400.0)
    corr = random_long_corridor(rng, [cls], min_km=mn, max_km=mx)
    launch_ll, target_ll = corr.sample_endpoints(rng)
    objects, dt, meta = _build_objects(cls, corr, rng, launch_ll, target_ll, evasive=True)
    prim = max([o for o in objects if o["type"] not in ("clutter",)], key=lambda o: len(o["P"]))
    P = np.asarray(prim["P"], float); V = np.asarray(prim["V"], float)
    alts = np.array([altitude_of(p) for p in P])
    ctrl = control_series_from_trajectory(P, V, dt, cls, alts, cmd_lat_accel=prim.get("cmd_lat_accel"),
                                          smooth_s=CUE_SMOOTH_S)
    lat = np.array([c.lat_accel_mps2 for c in ctrl])
    cue = np.array([c.lead_cue for c in ctrl])       # command-driven cue: ALWAYS the onset reference
    aoa = np.array([c.aoa_rad for c in ctrl])
    tt = np.arange(len(P)) * dt
    n = len(tt); need = max(1, int(round(0.5 / dt)))
    # TERMINAL maneuver onset (NOT the boost turn): the commanded-weave cue is ~0 in boost/cruise and
    # rises only at the terminal maneuver. Find the COMMAND onset there, then the KINEMATIC response
    # (lat crossing ONSET_G) at/after it -- which lands ~tau later. t_onset = the kinematic bend.
    if cue.max() <= 0:
        return None
    term = tt > 0.30 * tt[-1]                                   # skip boost/early cue blips
    cmd = np.where(term & (cue > 0.20 * cue.max()))[0]
    if len(cmd) == 0:
        return None
    c0 = int(cmd[0])
    kin = [i for i in range(c0, n - need) if (lat[i:i + need] > onset_g).all()]
    if not kin:
        return None
    if cue_mode == "kin":       # ablation: rebuild the SIGNATURE cue with no command channel
        ctrl_k = control_series_from_trajectory(P, V, dt, cls, alts, cmd_lat_accel=None,
                                               smooth_s=CUE_SMOOTH_S)
        cue_sig = np.array([c.lead_cue for c in ctrl_k])
    elif cue_mode == "off":     # ablation: no onset-related modulation in the signature at all
        cue_sig = np.zeros_like(cue)
    else:
        cue_sig = cue
    return dict(cls=cls, tt=tt, dt=dt, lat=lat, cue=cue_sig, aoa=aoa, V=V, P=P,
                t_onset=float(tt[kin[0]]), t_cmd=float(tt[c0]),
                tau=float(meta.get("actuator_lag_s", ACTUATOR_LAG_S)))


def measure(tr, noise_scale, fpr, seed, band=None, aspect_rad=0.8, kin_noise=0.0, return_raw=False):
    """aspect_rad: WAS HARDCODED 0.8 in five files, never swept, never reported.
    On identical tracks with byte-identical noise the paired delta runs +0.45 (p=8e-7) at 0.20 rad,
    +0.00 at 0.40, +0.15 at 0.80, +0.00 at 1.20, +0.10 at 1.50 -- an undeclared constant that moves
    the headline more than the headline. Default 0.8 reproduces every prior run; marginalize over it
    for any claim.
    kin_noise: the kinematic comparator thresholds NOISELESS truth, which is why
    it returns a constant +0.90 s at all five cue SNRs -- it is an oracle, not a detector. Adding
    m/s^2 of noise to the kinematic channel de-clairvoyances it. Default 0.0 = legacy oracle."""
    rng = np.random.default_rng(seed)
    tt, dt, lat, cue, aoa, V, t_on, tau = (tr[k] for k in
                                           ("tt", "dt", "lat", "cue", "aoa", "V", "t_onset", "tau"))
    lo, hi = band if band is not None else (_CUE_LO, _CUE_HI)
    vmag = np.linalg.norm(V, axis=1)
    cls = "marv" if tr["cls"] == "marv" else "supersonic_cruise"
    cru = (tt >= max(tt[0], t_on - 25.0)) & (tt <= t_on - 6.0)          # cruise window (pre-command) -> FPR thr
    if cru.sum() < 5:
        return None
    ci = np.where(cru)[0][::max(1, int(round(0.5 / dt)))]
    # FINE grid (0.1 s) for lead resolution. FIXED, and identical to the one kinematic_baselines.lead_of
    # uses for the comparator -- see EVAL_PRE/EVAL_POST.
    #
    # WAS: np.arange(t_on - max(4*tau + 1, 3), t_on + max(2*tau, 2.5), 0.1), i.e. a window whose width
    # was a function of the treatment being tested. That caps the largest reportable lead at a
    # tau-dependent value: 3.0 s for tau <= 0.5 and 5.8 s at tau = 1.2. Measured on the shipped cells,
    # 2-3 of 150 track-aspect observations per cell sit exactly on that cap. The bias is directional
    # and lands in the worst place: the excess at tau=1.2 differences a cell capped at 5.8 s against a
    # floor capped at 3.0 s, so a track truncated in the floor can report a larger lead in the
    # treatment cell purely because the window widened -- inflating the one contrast that carries the
    # paper's exception. A fixed window removes the treatment dependence; both arms now share it, so
    # the censoring that remains is common-mode and cancels in the paired difference.
    fg = np.arange(t_on - EVAL_PRE, t_on + EVAL_POST, 0.1)
    if len(fg) < 4:
        return None
    cue_f = resample(fg, tt, cue); lat_f = resample(fg, tt, lat)
    aoa_f = resample(fg, tt, aoa); v_f = resample(fg, tt, vmag)

    def md(cue_v, lat_v, aoa_v, v_v):
        c = Control(lat_accel_mps2=float(lat_v), lead_cue=float(cue_v), aoa_rad=float(aoa_v))
        # aspect_rad was HARDCODED 0.8 here while the function advertised it as a parameter, so
        # every "aspect sweep" silently produced N copies of the a=0.8 result. Caught 2026-07-25:
        # aspect_rad=-7.0 returned bit-identical output. Band power actually spans 27x over
        # 0.2-1.5 rad, so this is not cosmetic.
        spec = micro_doppler(cls, aspect_rad=aspect_rad, mach=float(v_v) / 343.0,
                             flight_state="cruise", control=c, rng=rng, noise_scale=noise_scale)
        return cue_band_power(spec, lo, hi)

    md_c = np.array([md(cue[i], lat[i], aoa[i], vmag[i]) for i in ci])            # cruise muD (cue ~ 0)
    md_o = np.array([md(cue_f[k], lat_f[k], aoa_f[k], v_f[k]) for k in range(len(fg))])
    lead_md = onset_anchored_lead(fg, md_o, float(np.quantile(md_c, 1 - fpr)), t_on)
    # kin_noise was declared and documented but never referenced here, so a caller asking for a
    # de-clairvoyanced comparator silently got the noiseless oracle back (kin_noise=1e6 returned a
    # bit-identical lead). Noise is applied to BOTH the observed channel and the cruise sample that
    # sets the threshold, so the matched-FPR protocol still holds.
    lat_obs, lat_cru = lat_f, lat[ci]
    if kin_noise > 0.0:
        lat_obs = lat_f + rng.normal(0.0, kin_noise, size=lat_f.shape)
        lat_cru = lat[ci] + rng.normal(0.0, kin_noise, size=lat[ci].shape)
    lead_kin = onset_anchored_lead(fg, lat_obs, float(np.quantile(lat_cru, 1 - fpr)), t_on)
    if not return_raw:
        return lead_md, lead_kin
    # return_raw: hand back the statistic itself, not just the decision. The threshold above is the
    # (1-fpr) quantile of THIS track's own pre-command cruise window -- per-track calibration in which
    # the evaluated track contributes to its own threshold. That is a different operating point from a
    # deployable detector with a fixed threshold, and an external review flagged it as leakage. Saving
    # md_c (calibration sample) and md_o (decision sample) lets a pooled or leave-one-track-out
    # threshold be applied afterwards, and the leads re-derived, WITHOUT re-rendering any spectrogram
    # and without disturbing the RNG stream -- so the self-calibrated and held-out numbers are
    # computed from bit-identical noise realisations and the difference is attributable to the
    # calibration change alone.
    return lead_md, lead_kin, md_c, md_o, fg, float(t_on)


def _boot_ci_median(a, rng, n_boot=10000, lo=2.5, hi=97.5):
    """Bootstrap percentile CI of the median -- descriptive percentiles are not inference."""
    if len(a) < 5:
        return float("nan"), float("nan")
    meds = np.median(rng.choice(a, size=(n_boot, len(a)), replace=True), axis=1)
    return float(np.percentile(meds, lo)), float(np.percentile(meds, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=30, help="tracks PER class")
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--onset-g", type=float, default=ONSET_G,
                    help="kinematic-departure threshold m/s^2 (sensitivity knob; default 2.0)")
    ap.add_argument("--cue-mode", choices=("cmd", "kin", "off"), default="cmd",
                    help="negative controls: kin = signature cue from kinematics only "
                         "(cannot lead; advantage should be <=0); off = no cue (detection should collapse)")
    ap.add_argument("--band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="cue band Hz (default 3-10; band_select.py justifies the default)")
    ap.add_argument("--out", default="runs/ml/snr_sweep.npz")
    args = ap.parse_args()
    noise_scales = [0.1, 0.3, 1.0, 3.0, 10.0]
    band = tuple(args.band) if args.band else None

    print("building maneuvering tracks (supersonic + marv, evasive=True); ONSET_G=%.1f m/s^2 | "
          "cue_mode=%s | band=%s..." % (args.onset_g, args.cue_mode,
                                        ("%.1f-%.1f Hz" % band) if band else "3-10 Hz default"), flush=True)
    tracks = []
    for cls in ("supersonic_cruise", "marv"):
        got = s = 0
        while got < args.tracks and s < args.tracks * 6:
            tr = build_track(cls, 5000 + (0 if cls == "supersonic_cruise" else 1000) + s,
                             onset_g=args.onset_g, cue_mode=args.cue_mode); s += 1
            if tr is not None:
                tracks.append(tr); got += 1
    tau = float(np.median([t["tau"] for t in tracks]))
    print("  %d tracks | tau = %.2f s (causal ceiling on lead)" % (len(tracks), tau), flush=True)

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    print("\n%-8s %-6s | %-30s | %-8s | %-11s | %s" %
          ("noise", "SNR", "muD ADVANTAGE over kinematic (s)", "muD>kin", "RAW muD lead", "pairs"))
    print("%-8s %-6s | %6s %6s %6s %5s | %8s | %5s %5s | %s" %
          ("scale", "dB", "med", "p25", "p75", "det", "frac", "med", "p75", "n"))
    print("-" * 90)
    rows, adv_all, cls_all, ns_idx_all, raw_all, trk_all = [], [], [], [], [], []
    boot_rng = np.random.default_rng(1234)
    for ns_i, ns in enumerate(noise_scales):
        advs, rawmd, clss, trks, wins, npair = [], [], [], [], 0, 0
        for k, tr in enumerate(tracks):
            r = measure(tr, ns, args.fpr, seed=9000 + k, band=band)
            if r is None:
                continue
            lm, lk = r
            if lm is None or lk is None:
                continue
            adv = lm - lk                                     # muD fires this many s BEFORE the kinematic detector
            advs.append(adv); rawmd.append(lm); clss.append(tr["cls"]); trks.append(k)
            npair += 1; wins += int(adv > 0)
        a = np.array(advs, float); rm = np.array(rawmd, float)   # rm = RAW muD lead vs onset (= tau + accel ramp)
        snr_db = -20.0 * np.log10(ns)
        _p = lambda q: (np.percentile(a, q) if len(a) else np.nan)
        _pr = lambda q: (np.percentile(rm, q) if len(rm) else np.nan)
        print("%-8.2f %-6.0f | %6.2f %6.2f %6.2f %4.0f%% | %8.2f | %5.2f %5.2f | %d" % (
            ns, snr_db, _p(50), _p(25), _p(75), 100 * len(a) / len(tracks),
            wins / max(npair, 1), _pr(50), _pr(75), npair))
        # -- per class, so pooling cannot hide a failing one, with the same interval --
        sub = []
        for c in ("supersonic_cruise", "marv"):
            ac = a[np.array(clss) == c] if len(a) else np.array([])
            sub.append("%s med %+.2f win %.0f%% n%d" % (
                "sup " if c == "supersonic_cruise" else "marv",
                (np.median(ac) if len(ac) else float("nan")),
                (100 * (ac > 0).mean() if len(ac) else 0), len(ac)))
        ci_lo, ci_hi = _boot_ci_median(a, boot_rng)
        if wilcoxon is not None and len(a) >= 10 and np.any(a != 0):
            try:
                p_w = float(wilcoxon(a, alternative="greater").pvalue)   # H1: median advantage > 0
            except ValueError:
                p_w = float("nan")
        else:
            p_w = float("nan")
        print("         %s | CI95[med] %+.2f..%+.2f | Wilcoxon(>0) p=%.4f" %
              (" | ".join(sub), ci_lo, ci_hi, p_w))
        adv_all.extend(advs); raw_all.extend(rawmd); trk_all.extend(trks)
        cls_all.extend(0 if c == "supersonic_cruise" else 1 for c in clss)
        ns_idx_all.extend([ns_i] * len(advs))
        rows.append(dict(noise_scale=ns, snr_db=snr_db, adv_med=float(_p(50)),
                         adv_p25=float(_p(25)), adv_p75=float(_p(75)),
                         rawmd_med=float(_pr(50)), rawmd_p75=float(_pr(75)),
                         win_frac=wins / max(npair, 1), npair=npair,
                         ci_lo=ci_lo, ci_hi=ci_hi, p_wilcoxon=p_w))
    print("\ntau (causal ceiling) = %.2f s. The muD ADVANTAGE (how much earlier muD fires than the"
          " kinematic\n  detector) should climb toward ~tau at high SNR and fall through 0 at the noise"
          " floor. It is\n  bounded by tau: muD cannot see the command more than tau before the kinematic"
          " response exists.\nHONESTY: surrogate signatures; modeled tau + cue; matched %.0f%% FPR." % (tau, 100*args.fpr))
    if args.cue_mode != "cmd":
        print("NEGATIVE CONTROL (cue_mode=%s): a positive median advantage here would mean the pipeline"
              "\n  manufactures lead without a command channel -> the cmd-mode result would be tautological."
              % args.cue_mode)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, tau=tau, fpr=args.fpr, cue_mode=args.cue_mode,
             onset_g=args.onset_g, noise_scales=np.array(noise_scales),
             snr_db=np.array([r["snr_db"] for r in rows]),
             adv_med=np.array([r["adv_med"] for r in rows]),
             adv_p25=np.array([r["adv_p25"] for r in rows]),
             adv_p75=np.array([r["adv_p75"] for r in rows]),
             rawmd_med=np.array([r["rawmd_med"] for r in rows]),
             win_frac=np.array([r["win_frac"] for r in rows]),
             ci_lo=np.array([r["ci_lo"] for r in rows]),
             ci_hi=np.array([r["ci_hi"] for r in rows]),
             p_wilcoxon=np.array([r["p_wilcoxon"] for r in rows]),
             adv_values=np.array(adv_all), adv_raw=np.array(raw_all),
             adv_cls=np.array(cls_all, dtype=int), adv_ns_idx=np.array(ns_idx_all, dtype=int),
             adv_track=np.array(trk_all, dtype=int))     # exact pairing for paired_delta.py
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
