"""experiments/slow_cue_budget.py -- can SNR or a shorter wavelength buy back the slow command?

THE SITUATION. The cue is z = trailing mean |dphi/dt|^2 with phi = (4*pi*r/lambda) sin(delta), so

    z_signal  ~  (r/lambda)^2 * (A/T)^2        A = fin excursion, T = command period
    z_noise   ~  set by the receiver, independent of T

A 15 s weave moves the fin ~60x slower than a step does, which costs ~35 dB of statistic. At the
shipped X-band (lambda = 3.0 cm) and 40 dB SNR the cue is buried and the detector never alarms
(shape_period_sweep, slow_cue_detector: six statistics tried, none recovers a clean null).

BUT lambda AND SNR ARE DESIGN CHOICES, NOT CONSTANTS OF THE PROBLEM. The 1/lambda^2 term says a
shorter carrier buys sensitivity directly:

    band    lambda     gain vs X-band
    X        3.00 cm      0 dB   (shipped)
    Ku       2.00 cm    +3.5 dB
    Ka       0.86 cm   +10.9 dB
    W        0.31 cm   +19.7 dB

So "the cue cannot convert a slow manoeuvre" may be a statement about ONE radar rather than about
the modality. This sweeps the two axes that actually pay for it -- carrier and SNR -- at the
generator's own command period, and asks whether any REALISABLE operating point converts.

The bar is the same one the boost-glide exclusion uses, so the negative and positive results are
judged by one rule: detection >= 50% at a NO-CUE NULL false-alarm rate < 5%.

    python experiments/slow_cue_budget.py --json runs/ml/slow_cue_budget.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AMP, DWELL, SIGMA, SEEDS, REPS = 0.2798, 0.002, 0.3, 30, 12
PERIOD = 15.0                       # the generator's own weave period for this class
BANDS = (("X", 10.0e9), ("Ka", 35.0e9), ("W", 95.0e9))
SNRS = (40.0, 55.0, 70.0, 85.0)


def run_cell(cell):
    band, fc, snr = cell
    import numpy as np
    from scipy.stats import wilcoxon
    import experiments.multiclass_lead as ml
    import experiments.dphi_sweep as dps
    import experiments.causal_dwell_test as cdt
    from experiments.multiclass_lead import class_windows
    from experiments.dphi_sweep import stat_matched_phase
    from experiments.causal_dwell_test import causal_lead, thr_from_cruise, DT_R, C

    # the carrier enters ONLY through lambda = C/FC; patch every binding that reads it
    cdt.FC_HZ = fc
    dps.FC_HZ = fc

    def return_at(t_fin, delta, snr_db, seed, fin_arm, shift_s=0.0, cue_on=True):
        rng = np.random.default_rng(seed)
        t = np.arange(t_fin[0], t_fin[-1], DT_R)
        d = (np.interp(t - shift_s, t_fin, delta, left=delta[0], right=delta[-1])
             if cue_on else np.zeros_like(t))
        lam = C / fc
        phase = 4.0 * np.pi * (fin_arm * np.sin(d)) / lam
        s = np.exp(1j * phase)
        p_n = 1.0 / (10.0 ** (snr_db / 10.0))
        return t, s + np.sqrt(p_n / 2.0) * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))

    dps.return_from_fin = return_at
    ml.return_from_fin = return_at
    ml.SHAPES = dict(ml.SHAPES)
    ml.SHAPES["supersonic_cruise"] = ("sinusoid", PERIOD)

    advs, mus, dets, fas = [], [], [], []
    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        if not wins:
            continue
        ea, em = [], []
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
            fl = ml.drive_airframe(w2["t"], w2["a_cmd"], w2["V"], w2["alt"])
            t_on, _ = ml.onset_from_achieved(fl["t"], fl["az"])
            if t_on is None:
                continue
            tf, delta, az = fl["t"], fl["delta"], fl["az"]
            for r in range(REPS):
                t, s = return_at(tf, delta, snr, 4000 + r, ml.FIN_ARM_M)
                z = stat_matched_phase(t, s, DWELL)
                lm = causal_lead(t, z, thr_from_cruise(t, z, t_on), t_on)
                rk = np.random.default_rng(4000 + r + 991)
                azr = np.abs(np.interp(t, tf, az) + rk.normal(0, SIGMA, len(t)))
                zk = ml.stat_cusum(azr, DWELL)
                lk = causal_lead(t, zk, thr_from_cruise(t, zk, t_on), t_on)
                dets.append(1.0 if lm is not None else 0.0)
                tn, sn = return_at(tf, delta, snr, 4000 + r, ml.FIN_ARM_M, cue_on=False)
                zn = stat_matched_phase(tn, sn, DWELL)
                fas.append(1.0 if causal_lead(tn, zn, thr_from_cruise(tn, zn, t_on), t_on)
                           is not None else 0.0)
                if lm is not None:
                    em.append(1000 * lm)
                if lm is not None and lk is not None:
                    ea.append(1000 * (lm - lk))
        if ea:
            advs.append(float(np.median(ea)))
        if em:
            mus.append(float(np.median(em)))

    a = np.asarray(advs, float)
    med = lambda v: float(np.median(v)) if len(v) else None                   # noqa: E731
    return dict(band=band, fc_hz=fc, snr_db=snr, period_s=PERIOD, n=int(a.size),
                adv_median=med(a), worst=float(a.min()) if a.size else None,
                n_pos=int((a > 0).sum()), muD_lead_ms=med(mus),
                det=float(np.mean(dets)) if dets else None,
                fa=float(np.mean(fas)) if fas else None,
                p=float(wilcoxon(a).pvalue) if a.size >= 3 and np.any(a != 0) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--procs", type=int, default=max(1, min(12, cpu_count() - 2)))
    args = ap.parse_args()

    print("SLOW-CUE LINK BUDGET -- carrier x SNR at the generator's own %.0f s command period" % PERIOD)
    print("  the shipped point is X-band / 40 dB, where the cue does not alarm at all.")
    print("  bar: detection >=50%% at a NO-CUE NULL FA <5%% (same rule as the boost-glide exclusion)\n")
    jobs = [(b, f, s) for (b, f) in BANDS for s in SNRS]
    with Pool(args.procs, maxtasksperchild=1) as pool:
        rows = pool.map(run_cell, jobs, chunksize=1)

    print("%5s %8s %6s %5s %11s %9s %6s %7s %6s"
          % ("band", "lambda", "SNR", "n", "advantage", "worst", "pos", "det", "null"))
    print("-" * 74)
    for r in rows:
        f = lambda v, q="%+.2f": (q % v) if v is not None else "--"           # noqa: E731
        print("%5s %7.2fcm %5.0f %5d %11s %9s %3d/%-2d %6.0f%% %5.0f%%"
              % (r["band"], 100 * 2.998e8 / r["fc_hz"], r["snr_db"], r["n"],
                 f(r["adv_median"]), f(r["worst"], "%+.1f"), r["n_pos"], r["n"],
                 100 * (r["det"] or 0), 100 * (r["fa"] or 0)))

    # TWO SEPARATE QUESTIONS, and conflating them is how this script first reported a success on a
    # negative result. DETECTABLE = the cue clears the null. CONVERTING = it also beats the
    # comparator. A link budget buys the first and cannot buy the second.
    seen = [r for r in rows if (r["det"] or 0) >= 0.5
            and (r["fa"] if r["fa"] is not None else 1.0) < 0.05]
    won = [r for r in seen if (r["adv_median"] or -1.0) > 0 and r["n_pos"] > 0]
    print("\nDETECTABLE (det>=50%%, null FA<5%%): %s"
          % (", ".join("%s/%.0f dB" % (r["band"], r["snr_db"]) for r in seen) or "NONE"))
    print("CONVERTING  (also beats the comparator): %s"
          % (", ".join("%s/%.0f dB" % (r["band"], r["snr_db"]) for r in won) or "NONE"))
    if seen and not won:
        best = min(seen, key=lambda r: (r["snr_db"], r["fc_hz"]))
        print("\n=> DETECTABILITY is a link budget -- %s at %.0f dB clears the null at %.0f%% "
              "detection." % (best["band"], best["snr_db"], 100 * (best["det"] or 0)))
        print("   THE ADVANTAGE IS NOT. At this command period the muD arm alarms LATER than the")
        print("   comparator at every point tested (%d of %d trajectories positive, best median "
              "%+.1f ms)." % (sum(r["n_pos"] for r in seen), sum(r["n"] for r in seen),
                              max(r["adv_median"] for r in seen if r["adv_median"] is not None)))
        print("   That is the physical claim the letter should make: the muD edge is a FAST-")
        print("   TRANSIENT phenomenon. A slow manoeuvre is exactly what a state-based detector")
        print("   integrates well, so buying SNR makes the cue visible without making it early.")
    elif won:
        print("\n=> A realisable point both detects AND beats the comparator; re-anchor the letter.")
    else:
        print("\n=> Not even detectable within a 45 dB SNR sweep and a 10x wavelength reduction.")
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(period_s=PERIOD, sigma=SIGMA, seeds=SEEDS, reps=REPS, amp_factor=AMP,
                       bands=[b for b, _ in BANDS], snrs=list(SNRS), rows=rows),
                  open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
