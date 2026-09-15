"""experiments/physical_cue_test.py -- micro-Doppler cue driven by observable body states.

A radar observes the body, and the guidance command is not directly observable. A muD cue driven
from `cmd_lat_accel`, raced against a kinematic truth equal to the same command delayed by tau,
measures tau: the advantage regresses on tau at r^2 = 0.984 and persists with the radar chain
removed.

This script propagates the command through the airframe and servo model (sim/sixdof.py) and drives
the cue from body states a radar can observe:

    delta(t)  fin deflection      -- the control-surface observable used in the paper
    q(t)      body pitch rate     -- 437 Hz = 28 Doppler bins during a 9.1 g command, 39x the
                                     modelled fin cue, peaking at 112 ms
    alpha(t)  angle of attack     -- tilts the body axis off the velocity vector

The kinematic detector reads the achieved lateral acceleration a_z(t). The command does not enter
the muD path, so any lead measured here comes from body motion.

The window of interest is 0.03-0.17 s. Trajectories sampled at dt = 0.5 s and measured on a 0.1 s
grid cannot resolve it, so this script integrates at dt = 1e-4 s and detects on a 1 ms grid.

    python experiments/physical_cue_test.py
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
from sim.sixdof import PitchAirframe, DEFAULT_AIRFRAME, G0, sound_speed       # noqa: E402
from sim.signatures import micro_doppler, Control                             # noqa: E402
from experiments.onset_snr_sweep import cue_band_power, onset_anchored_lead   # noqa: E402

DT = 1e-4                      # integration step (s); the window is tens of ms
GRID = 1e-3                    # detection grid (s)
ALT, MACH = 25000.0, 7.5       # the paper's supersonic_cruise band
ONSET_G = 2.0                  # kinematic departure threshold (m/s^2), as in the paper
FPR = 0.10


def fly(a_cmd_g, t_cmd, t_end, alt=ALT, mach=MACH, airframe=None):
    """Propagate a commanded lateral-acceleration step of a_cmd_g (g), applied at t_cmd (s),
    through the airframe and servo model. Returns a dict with the observable body states
    (delta, q, alpha), the achieved lateral acceleration az, the time base t, V and t_cmd."""
    V = mach * sound_speed(alt)
    af = PitchAirframe(airframe or DEFAULT_AIRFRAME)
    n = int(t_end / DT)
    t = np.arange(n) * DT
    delta = np.zeros(n); q = np.zeros(n); alpha = np.zeros(n); az = np.zeros(n)
    for i in range(n):
        cmd = (a_cmd_g * G0) if t[i] >= t_cmd else 0.0
        az[i] = af.step(cmd, V, alt, DT)
        delta[i] = af.delta; q[i] = af.q; alpha[i] = af.alpha
    return dict(t=t, delta=delta, q=q, alpha=alpha, az=az, V=V, t_cmd=t_cmd)


def _norm(x, ref_win):
    """Normalise |x| to a 0..1 cue by its 99th percentile over the whole record. Returns
    (cue, standard deviation of |x| over ref_win)."""
    base = np.abs(x[ref_win])
    scale = max(float(np.percentile(np.abs(x), 99)), 1e-12)
    return np.clip(np.abs(x) / scale, 0.0, 1.0), float(np.std(base))


def detect(t, cue, az, alpha, V, t_on, seed, noise_scale, cue_name):
    """Render the muD spectrum from a body-state cue and threshold its cue-band power at the
    matched FPR set on the pre-command window. Returns the onset-anchored lead (s), or None.
    The command does not enter this function."""
    rng = np.random.default_rng(seed)
    cru = (t >= t_on - 0.60) & (t <= t_on - 0.10)          # pre-command window sets the threshold
    grid = np.arange(t_on - 0.40, t_on + 0.40, GRID)
    if cru.sum() < 8 or len(grid) < 8:
        return None

    def bp(idx_t):
        i = int(np.searchsorted(t, idx_t))
        i = min(max(i, 0), len(t) - 1)
        c = Control(lat_accel_mps2=float(abs(az[i])), lead_cue=float(cue[i]),
                    aoa_rad=float(abs(alpha[i])))
        spec = micro_doppler("supersonic_cruise", aspect_rad=0.8, mach=float(V) / 343.0,
                             flight_state="cruise", control=c, rng=rng, noise_scale=noise_scale)
        return cue_band_power(spec)

    cru_t = t[cru][:: max(1, int(0.01 / DT))]
    md_c = np.array([bp(x) for x in cru_t])
    md_o = np.array([bp(x) for x in grid])
    thr = float(np.quantile(md_c, 1 - FPR))
    return onset_anchored_lead(grid, md_o, thr, t_on, need=2)


def kinematic_lead(t, az, t_on, seed, kin_noise):
    """Kinematic detector that thresholds the noisy achieved lateral acceleration at the matched
    FPR. Returns the onset-anchored lead (s), or None."""
    rng = np.random.default_rng(seed + 991)
    cru = (t >= t_on - 0.60) & (t <= t_on - 0.10)
    grid = np.arange(t_on - 0.40, t_on + 0.40, GRID)
    obs = np.interp(grid, t, az) + rng.normal(0, kin_noise, len(grid))
    cru_s = az[cru][:: max(1, int(0.01 / DT))] + rng.normal(0, kin_noise, len(az[cru][:: max(1, int(0.01 / DT))]))
    thr = float(np.quantile(np.abs(cru_s), 1 - FPR))
    return onset_anchored_lead(grid, np.abs(obs), thr, t_on, need=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24, help="noise realisations")
    ap.add_argument("--a-cmd", type=float, default=9.1, help="commanded lateral accel (g)")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.1, 1.0],
                    help="muD noise scales (+20 dB, 0 dB)")
    ap.add_argument("--kin-noise", type=float, default=0.3,
                    help="kinematic-channel noise (m/s^2); 0 gives a noiseless oracle")
    args = ap.parse_args()

    t_cmd = 1.0
    fl = fly(args.a_cmd, t_cmd, t_cmd + 1.5)
    t, az = fl["t"], fl["az"]

    # Onset reference: first crossing of ONSET_G by the achieved acceleration held for 5 ms.
    above = np.abs(az) > ONSET_G
    need = int(0.005 / DT)
    idx = [i for i in range(len(t) - need) if above[i:i + need].all()]
    t_on = float(t[idx[0]]) if idx else float("nan")

    print("Physical-cue test: muD driven by body states (fin, pitch rate, angle of attack)")
    print("  %.0f km / M%.1f / %.1f g step at t=%.2f s | airframe+servo at dt=%g s" %
          (ALT / 1000, MACH, args.a_cmd, t_cmd, DT))
    print("  kinematic onset (|a_z| > %.1f m/s^2 sustained): t=%.4f s  => %.1f ms after command"
          % (ONSET_G, t_on, 1000 * (t_on - t_cmd)))
    print("  peak fin %.3f deg | peak pitch rate %.3f deg/s | peak a_z %.1f m/s^2"
          % (np.degrees(np.abs(fl["delta"]).max()), np.degrees(np.abs(fl["q"]).max()),
             np.abs(az).max()))
    print("\n  THE CEILING: no body-state detector can beat %.1f ms, because that is when the"
          % (1000 * (t_on - t_cmd)))
    print("  achieved acceleration crosses the kinematic threshold. A larger lead is an error.\n")

    ref = (t >= t_cmd - 0.5) & (t < t_cmd)
    cues = {}
    for name, x in (("fin delta(t)", fl["delta"]), ("pitch rate q(t)", fl["q"]),
                    ("AoA alpha(t)", fl["alpha"])):
        c, base = _norm(x, ref)
        cues[name] = c

    print("%-18s %-9s %10s %10s %10s %9s" %
          ("cue source", "muD SNR", "muD lead", "kin lead", "ADVANTAGE", "muD wins"))
    print("-" * 74)
    for ns, db in zip(args.noise, ("+20 dB", "0 dB")):
        for name, cue in cues.items():
            mds, kins, adv = [], [], []
            for r in range(args.reps):
                lm = detect(t, cue, az, fl["alpha"], fl["V"], t_on, 5000 + r, ns, name)
                lk = kinematic_lead(t, az, t_on, 5000 + r, args.kin_noise)
                if lm is not None and lk is not None:
                    mds.append(lm); kins.append(lk); adv.append(lm - lk)
            if not adv:
                print("%-18s %-9s %10s" % (name, db, "no detect")); continue
            a = np.array(adv)
            print("%-18s %-9s %+9.1f ms %+9.1f ms %+9.1f ms %8.0f%%" %
                  (name, db, 1000 * np.median(mds), 1000 * np.median(kins),
                   1000 * np.median(a), 100 * (a > 0).mean()))
    print("\nADVANTAGE > 0 means a cue built only from observable body motion fires before the")
    print("  kinematic detector on the same event. ADVANTAGE <= 0 means it fires at the same time")
    print("  or later.")
    print("  kinematic channel noise = %.2f m/s^2 (0 would restore the noiseless oracle)."
          % args.kin_noise)


if __name__ == "__main__":
    main()
