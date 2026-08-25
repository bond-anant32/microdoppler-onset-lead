"""experiments/physical_cue_test.py -- does the micro-Doppler cue anticipate the trajectory?

WHAT THIS MEASURES. The guidance command is not an observable: a radar sees the body, not the
command. Driving the muD cue from `cmd_lat_accel` against a kinematic truth that is the same
command delayed by tau measures tau, not the channel -- the advantage then regresses on tau at
r^2 = 0.984 and survives deleting the entire radar chain.

Here the command is propagated through the airframe and servo (sim/sixdof.py) and the cue is
driven by BODY STATES a radar could observe:

    delta(t)  fin deflection      -- the control-surface observable the letter uses
    q(t)      body pitch rate     -- 437 Hz = 28 Doppler bins during a 9.1 g command, 39x the
                                     modelled fin cue, peaking at 112 ms
    alpha(t)  angle of attack     -- tilts the body axis off the velocity vector

against a kinematic detector reading the achieved lateral acceleration a_z(t). Nothing in the muD
path touches the command. If a lead survives here it is physical; if it does not, the hypothesis is
dead on its own terms.

RESOLUTION. The window in question is 0.03-0.17 s. Trajectories sampled at dt = 0.5 s and measured
on a 0.1 s grid cannot resolve it by construction, so everything here runs at dt = 1e-4 s and
measures on a 1 ms grid.

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

DT = 1e-4                      # integration step (s) -- the window is tens of ms
GRID = 1e-3                    # detection grid (s)
ALT, MACH = 25000.0, 7.5       # the paper's supersonic_cruise band
ONSET_G = 2.0                  # kinematic departure threshold (m/s^2), as in the paper
FPR = 0.10


def fly(a_cmd_g, t_cmd, t_end, alt=ALT, mach=MACH, airframe=None):
    """Propagate a commanded lateral-accel step through the REAL airframe + servo.
    Returns the body states a radar could observe, plus the kinematic truth."""
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
    """Normalise a body state to a 0..1 cue by its own pre-command variability, NOT by its own 95th
    percentile of the whole record (which is what let the command cue saturate to a 0->1 step)."""
    base = np.abs(x[ref_win])
    scale = max(float(np.percentile(np.abs(x), 99)), 1e-12)
    return np.clip(np.abs(x) / scale, 0.0, 1.0), float(np.std(base))


def detect(t, cue, az, alpha, V, t_on, seed, noise_scale, cue_name):
    """Render muD from a BODY-STATE cue and threshold its band power at matched FPR.
    Returns the onset-anchored lead. The command never enters this function."""
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
    """Deployable kinematic detector: threshold NOISY achieved lateral accel at matched FPR."""
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
                    help="kinematic-channel noise (m/s^2); 0 = the old noiseless oracle")
    args = ap.parse_args()

    t_cmd = 1.0
    fl = fly(args.a_cmd, t_cmd, t_cmd + 1.5)
    t, az = fl["t"], fl["az"]

    # PHYSICAL onset reference: first sustained crossing of ONSET_G by the ACHIEVED accel.
    above = np.abs(az) > ONSET_G
    need = int(0.005 / DT)
    idx = [i for i in range(len(t) - need) if above[i:i + need].all()]
    t_on = float(t[idx[0]]) if idx else float("nan")

    print("PHYSICAL-CUE TEST -- muD driven by BODY STATES, never by the command")
    print("  %.0f km / M%.1f / %.1f g step at t=%.2f s | airframe+servo at dt=%g s" %
          (ALT / 1000, MACH, args.a_cmd, t_cmd, DT))
    print("  kinematic onset (|a_z| > %.1f m/s^2 sustained): t=%.4f s  => %.1f ms after command"
          % (ONSET_G, t_on, 1000 * (t_on - t_cmd)))
    print("  peak fin %.3f deg | peak pitch rate %.3f deg/s | peak a_z %.1f m/s^2"
          % (np.degrees(np.abs(fl["delta"]).max()), np.degrees(np.abs(fl["q"]).max()),
             np.abs(az).max()))
    print("\n  THE CEILING: no body-state detector can beat %.1f ms, because that is when the"
          % (1000 * (t_on - t_cmd)))
    print("  achieved acceleration crosses the kinematic threshold. Anything larger is a bug.\n")

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
    print("\nREADING: ADVANTAGE > 0 means a cue built ONLY from observable body motion fires before a")
    print("  kinematic detector on the same event. That is the hypothesis, tested without the")
    print("  command channel. ADVANTAGE <= 0 kills it on its own terms.")
    print("  kinematic channel noise = %.2f m/s^2 (0 would restore the noiseless oracle)."
          % args.kin_noise)


if __name__ == "__main__":
    main()
