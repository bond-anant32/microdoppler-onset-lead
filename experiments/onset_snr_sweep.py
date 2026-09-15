"""Onset lead of a micro-Doppler detector and a kinematic detector as a function of cue SNR.

In the lagged-autopilot model the commanded body-pitch/control response leads the velocity-vector
turn by tau (ACTUATOR_LAG_S = 1.2 s). The micro-Doppler (muD) statistic reads the command and the
kinematic statistic reads the response.

Leads are measured against t_onset, the first sustained crossing of the lateral-acceleration
threshold ONSET_G. The crossing lags the physical response by the acceleration ramp, so raw muD
leads can exceed tau (lead = tau + ramp); the tau bound applies against the response onset
t_cmd + tau. At loose FPR the kinematic detector can also lead the threshold crossing slightly.
The causal quantity is the paired per-track difference against a cue-severed control (--cue-mode).

  * Both detectors' leads are measured from the same fixed t_onset, so a late kinematic alarm cannot
    inflate the muD lead.
  * The alarm is onset-anchored: the contiguous above-threshold run leading into t_onset, tolerating
    one dropout. A cruise false alarm far from onset does not register as lead, and the lead is
    bounded by the time the cue physically rises.
  * muD statistic: 3-10 Hz band power of the spectrogram's per-column energy. The 4-8 Hz
    pitch/flutter modulation is narrower than one Doppler bin, so it appears as a slow-time
    oscillation of column energy. Kinematic statistic: |lateral acceleration| (SNR-independent).
  * Each spectrogram is rendered from the instantaneous state (zero dwell-fill latency), so absolute
    leads are upper bounds.
  * 30 tracks per class; IQR, bootstrap CI of the median and a one-sided Wilcoxon test are reported.

Signatures are physics-based surrogates. tau and the cue model are calibrated model parameters.

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
ONSET_G = 2.0                                     # m/s^2 (~0.2 g), kinematic-departure threshold
# Cue-filter width (s) passed to control_series_from_trajectory, set by the CUE_SMOOTH_S environment
# variable. The filter length is k = max(3, int(smooth_s / dt)) samples, so at dt = 0.5 s the default
# 1.0 gives a 3-sample (1.5 s) causal boxcar. That is long compared with the 0.15-0.45 s lead expected
# under first-order lag and with the ~12 ms deflection time of a real fin.
CUE_SMOOTH_S = float(os.environ.get("CUE_SMOOTH_S", "1.0"))

# Resampling of a trajectory-grid series onto the 0.1 s detection grid, set by the RESAMPLE
# environment variable.
#   'causal'  previous-sample hold: the value at t is the last trajectory sample at or before t.
#   'linear'  np.interp.
# The cue is a step. control_series_from_trajectory normalises by ref = max(percentile(src, 95), 1.0);
# the weave occupies ~3% of the record, so the 95th percentile of |cmd_lat_accel| is 0, ref is 1.0
# against a commanded peak of 92-138 m/s^2, and lead_cue saturates to 1 within one sample. Linear
# interpolation spreads that step backwards across one trajectory sample (0.5 s). At ACTUATOR_LAG_S = 0,
# where t_onset == t_cmd on all 30 tracks, 'linear' places the muD alarm a median 0.55 s before the
# command on 30 of 30 tracks. Relative to 'linear', 'causal' shifts leads by exactly 0.50 s (one
# trajectory sample) on 29 of 30 tracks, at every tau and in both command models. Because the shift is
# a constant offset, differences against the tau = 0 floor are unaffected; the floor and all absolute
# leads move by 0.50 s.
RESAMPLE = os.environ.get("RESAMPLE", "causal").lower()

# Detection-grid extent relative to t_onset, shared by every detector arm and independent of tau.
# EVAL_PRE abuts the cruise calibration window, which ends at t_onset - 6.0 s, so evaluation and
# calibration never overlap.
EVAL_PRE, EVAL_POST = 6.0, 3.0


def resample(fg, tt, y, how=None):
    """Resample the trajectory-grid series y(tt) onto the detection grid fg (see RESAMPLE).

    'causal' uses no future samples."""
    if (how or RESAMPLE) == "linear":
        return np.interp(fg, tt, y)
    idx = np.searchsorted(np.asarray(tt), np.asarray(fg), side="right") - 1
    return np.asarray(y)[np.clip(idx, 0, len(y) - 1)]


def cue_band_power(spec, lo=_CUE_LO, hi=_CUE_HI):
    """muD statistic: band power of the spectrogram's per-column energy in [lo, hi] Hz (default 3-10).

    The power is divided by N * mean(energy)^2, which makes it insensitive to the per-spectrogram
    max normalisation. Returns 0.0 for a near-empty spectrogram or fewer than 8 columns."""
    ce = spec.sum(axis=0).astype(float); mu = ce.mean()
    if mu < 1e-9 or len(ce) < 8:
        return 0.0
    P = np.abs(np.fft.rfft(ce - mu)) ** 2
    f = np.fft.rfftfreq(len(ce), d=_COL_DT)
    return float(P[(f >= lo) & (f <= hi)].sum() / (mu * mu * len(ce)))


def loo_leads(cal, dec, fpr=None, rel=None):
    """Leave-one-track-out threshold applied to a cell's saved calibration and decision traces.

    The threshold for track i is the (1 - fpr) quantile of the cruise samples pooled over all other
    tracks of the same cell, so no track contributes to its own threshold. The same rule applies to
    both detector arms.

      cal  (n, Kc)  per-track cruise calibration samples, NaN-padded to a common width
      dec  (n, Kd)  per-track decision samples on the fixed 0.1 s grid
      fpr           false-alarm rate (default 0.10)
      rel           the grid `dec` is on, relative to t_onset (default: the shared EVAL_PRE/EVAL_POST window)

    Returns an (n,) array of leads, NaN where the track raised no scoreable alarm. No spectrogram is
    re-rendered and no RNG is consumed, so the result differs from the per-track threshold in
    measure() only through the calibration.
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
    """Lead of the onset-anchored sustained alarm.

    The alarm run is the contiguous run of above-threshold samples leading into t_onset, walked back
    tolerating up to max_gap dropouts, and lead = t_onset - t_run_start (> 0 is early). If the
    statistic is below threshold at onset, the first sustained alarm within late_tol after onset
    gives a negative lead. Returns None when the track is missed."""
    above = stat > thr
    n = len(t)
    j = int(np.searchsorted(t, t_onset, side="right")) - 1
    j = min(max(j, 0), n - 1)
    # `need` is the sustain requirement, enforced on both paths so that a single-sample threshold
    # blip cannot count as an alarm: the run must contain >= need above-threshold samples.
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
    # A run that begins at onset and continues forward also counts as sustained. The forward
    # continuation is counted before demotion, so a {j, j+1} run of exactly `need` samples spanning
    # onset is kept; the late path would miss it.
    kf = j
    while run_len < need and kf + 1 < n and above[kf + 1] and t[kf + 1] <= t_onset + late_tol:
        run_len += 1; kf += 1
    if run_len < need:                                   # onset-anchored run is a blip -> late path
        fwd = [k2 for k2 in range(j + 1, n) if t[k2] <= t_onset + late_tol and above[k2]
               and sum(above[k2:min(k2 + need, n)]) >= min(need, n - k2)]
        return (t_onset - t[fwd[0]]) if fwd else None
    return t_onset - t[start]


def build_track(cls, seed, onset_g=ONSET_G, cue_mode="cmd"):
    """Build one maneuvering track of class cls and locate its command and kinematic onsets.

    cue_mode selects the signature cue:
      'cmd'  the command channel (lagged-autopilot model; leads the response by tau).
      'kin'  the causal-smoothed kinematic envelope, with no command knowledge. The muD statistic
             sees a maneuver modulation that cannot lead, so the expected advantage is <= 0.
      'off'  zero cue with kinematics unchanged. Detection is expected to fall to the FPR floor.
    t_cmd and t_onset are always computed from the command channel, so all three modes share one
    onset reference. Returns a dict of track series and onset times, or None if no terminal onset
    is found."""
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
    cue = np.array([c.lead_cue for c in ctrl])       # command-driven cue, always the onset reference
    aoa = np.array([c.aoa_rad for c in ctrl])
    tt = np.arange(len(P)) * dt
    n = len(tt); need = max(1, int(round(0.5 / dt)))
    # Terminal maneuver onset. The commanded-weave cue is ~0 in boost and cruise and rises at the
    # terminal maneuver. t_cmd is the command onset there; t_onset is the first sustained crossing of
    # onset_g by the lateral acceleration at or after t_cmd, about tau later.
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
    if cue_mode == "kin":       # ablation: rebuild the signature cue with no command channel
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
    """Leads of the muD and kinematic detectors on one track at one noise scale.

      noise_scale  micro-Doppler noise scale (SNR dB = -20 log10(noise_scale))
      fpr          false-alarm rate; each threshold is the (1 - fpr) quantile of the track's cruise window
      band         (lo, hi) cue band in Hz (default 3-10)
      aspect_rad   aspect angle passed to micro_doppler (default 0.8). On identical tracks and noise
                   the paired delta is +0.45 (p = 8e-7) at 0.20 rad, +0.00 at 0.40, +0.15 at 0.80,
                   +0.00 at 1.20 and +0.10 at 1.50.
      kin_noise    standard deviation (m/s^2) of Gaussian noise added to the kinematic channel. At the
                   default 0.0 the kinematic detector thresholds noiseless truth and acts as an oracle,
                   giving a constant +0.90 s lead at all five cue SNRs.
      return_raw   also return the muD calibration and decision samples, the grid and t_onset

    Returns (lead_md, lead_kin), or (lead_md, lead_kin, md_c, md_o, fg, t_onset) with return_raw.
    A lead is None when that detector misses; the function returns None if the cruise window or
    detection grid is too short."""
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
    # Fine 0.1 s grid for lead resolution, shared with the kinematic comparator (see EVAL_PRE/EVAL_POST).
    # Its extent is fixed relative to t_on and independent of tau, so any censoring of long leads is
    # common to both arms and cancels in the paired difference.
    fg = np.arange(t_on - EVAL_PRE, t_on + EVAL_POST, 0.1)
    if len(fg) < 4:
        return None
    cue_f = resample(fg, tt, cue); lat_f = resample(fg, tt, lat)
    aoa_f = resample(fg, tt, aoa); v_f = resample(fg, tt, vmag)

    def md(cue_v, lat_v, aoa_v, v_v):
        c = Control(lat_accel_mps2=float(lat_v), lead_cue=float(cue_v), aoa_rad=float(aoa_v))
        # Cue band power varies by a factor of 27 over aspect 0.2-1.5 rad.
        spec = micro_doppler(cls, aspect_rad=aspect_rad, mach=float(v_v) / 343.0,
                             flight_state="cruise", control=c, rng=rng, noise_scale=noise_scale)
        return cue_band_power(spec, lo, hi)

    md_c = np.array([md(cue[i], lat[i], aoa[i], vmag[i]) for i in ci])            # cruise muD (cue ~ 0)
    md_o = np.array([md(cue_f[k], lat_f[k], aoa_f[k], v_f[k]) for k in range(len(fg))])
    lead_md = onset_anchored_lead(fg, md_o, float(np.quantile(md_c, 1 - fpr)), t_on)
    # kin_noise is added to both the observed channel and the cruise sample that sets the threshold,
    # so the kinematic threshold stays matched in FPR.
    lat_obs, lat_cru = lat_f, lat[ci]
    if kin_noise > 0.0:
        lat_obs = lat_f + rng.normal(0.0, kin_noise, size=lat_f.shape)
        lat_cru = lat[ci] + rng.normal(0.0, kin_noise, size=lat[ci].shape)
    lead_kin = onset_anchored_lead(fg, lat_obs, float(np.quantile(lat_cru, 1 - fpr)), t_on)
    if not return_raw:
        return lead_md, lead_kin
    # The thresholds above are the (1 - fpr) quantile of this track's own pre-command cruise window,
    # so the evaluated track contributes to its own threshold. With return_raw, md_c (calibration
    # sample) and md_o (decision sample) are returned so that a pooled or leave-one-track-out
    # threshold (loo_leads) can be applied afterwards on the same noise realisations, with no
    # spectrogram re-rendered.
    return lead_md, lead_kin, md_c, md_o, fg, float(t_on)


def _boot_ci_median(a, rng, n_boot=10000, lo=2.5, hi=97.5):
    """Bootstrap percentile CI of the median (default 2.5-97.5 percentiles, 10000 resamples).

    Returns (nan, nan) for fewer than 5 samples."""
    if len(a) < 5:
        return float("nan"), float("nan")
    meds = np.median(rng.choice(a, size=(n_boot, len(a)), replace=True), axis=1)
    return float(np.percentile(meds, lo)), float(np.percentile(meds, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=30, help="tracks per class")
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--onset-g", type=float, default=ONSET_G,
                    help="kinematic-departure threshold in m/s^2 (default 2.0)")
    ap.add_argument("--cue-mode", choices=("cmd", "kin", "off"), default="cmd",
                    help="signature cue: cmd = command channel; kin = kinematics only (negative control, "
                         "expected advantage <= 0); off = no cue (negative control, detection at the FPR floor)")
    ap.add_argument("--band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="cue band in Hz (default 3-10)")
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
            adv = lm - lk                                     # seconds by which muD alarms before the kinematic detector
            advs.append(adv); rawmd.append(lm); clss.append(tr["cls"]); trks.append(k)
            npair += 1; wins += int(adv > 0)
        a = np.array(advs, float); rm = np.array(rawmd, float)   # rm = raw muD lead vs onset (tau + accel ramp)
        snr_db = -20.0 * np.log10(ns)
        _p = lambda q: (np.percentile(a, q) if len(a) else np.nan)
        _pr = lambda q: (np.percentile(rm, q) if len(rm) else np.nan)
        print("%-8.2f %-6.0f | %6.2f %6.2f %6.2f %4.0f%% | %8.2f | %5.2f %5.2f | %d" % (
            ns, snr_db, _p(50), _p(25), _p(75), 100 * len(a) / len(tracks),
            wins / max(npair, 1), _pr(50), _pr(75), npair))
        # per-class median advantage, win rate and count
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
          " response exists.\nModel: surrogate signatures; modeled tau and cue; matched %.0f%% FPR." % (tau, 100*args.fpr))
    if args.cue_mode != "cmd":
        print("Negative control (cue_mode=%s): with no command channel the median advantage"
              "\n  is expected to be <= 0."
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
             adv_track=np.array(trk_all, dtype=int))     # track index, for per-track pairing across runs
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
