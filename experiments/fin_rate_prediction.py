"""experiments/fin_rate_prediction.py -- test Sec. III's design law on an axis it was never fit to.

THE CLAIM UNDER TEST. Section III states a design law: detection is governed by peak |dphi/dt|,
which is a RATE, and there is a usable/unusable transition bracketed in [329, 412] rad/s. That
bracket comes from runs/ml/dphi_ladder.json, which was produced by varying ONE thing -- the lever
arm r -- at a fixed airframe. peak |dphi/dt| = (4*pi/lambda) * r * max|cos(delta) * delta_dot|, so
sweeping r sweeps the peak while leaving the fin motion untouched. A law fitted that way is a law
about r, and calling it a law about RATE is an extrapolation.

WHAT MAKES THIS A REAL TEST. experiments/airframe_class_sweep.py moves the OTHER factor: same lever
arm, same carrier, same command, but six airframes whose closed-loop response spans a decade, so
max|delta_dot| moves and r does not. If the design law is what Sec. III says it is, the SAME
bracket must separate the converting airframes from the non-converting ones -- a prediction on an
axis with no free parameters left to fit.

PREDICTION, STATED BEFORE RUNNING AND FALSIFIABLE EITHER WAY:

    every airframe that converts        has median peak |dphi/dt| >= 411.6 rad/s   (ladder's hi)
    every airframe that does not        has median peak |dphi/dt| <= 329.3 rad/s   (ladder's lo)
    and the two groups do not interleave

If it holds, Sec. III's sentence is a mechanism rather than a curve fit, and the airframe boundary
in Sec. IV has the same cause as the command-duration boundary -- which is what the letter asserts.
If it FAILS, the letter is asserting a common cause it has not got, and that sentence has to be
rewritten. A near miss (one airframe inside the bracket) is a THIRD outcome and is reported as
such, not rounded to either.

The peak is computed through the SAME code path the measurement uses -- return_from_fin at an SNR
high enough that the noise term is negligible, then dphi() -- rather than from the closed form, so
a discrepancy in the renderer cannot hide behind agreeing algebra.

    python experiments/fin_rate_prediction.py --json runs/ml/fin_rate_prediction.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.airframe_class_sweep import AIRFRAMES, _install, AMP, SEEDS   # noqa: E402

QUIET_SNR_DB = 200.0      # noise power 1e-20 -- present in the path, negligible in the peak


def peak_dphi(airframe, seeds=SEEDS):
    """Median and range of peak |dphi/dt| over the flown trajectories, for one airframe."""
    _install(airframe)
    from experiments.multiclass_lead import class_windows, drive_airframe, FIN_ARM_M
    from experiments.dphi_sweep import return_from_fin, dphi

    peaks, rates = [], []
    for sd in range(seeds):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        for w in wins:
            cmd = np.asarray(w["a_cmd"], float) * AMP
            fl = drive_airframe(w["t"], cmd, w["V"], w["alt"])
            _t, s = return_from_fin(fl["t"], fl["delta"], QUIET_SNR_DB, 4000, FIN_ARM_M)
            peaks.append(float(np.max(dphi(s))))
            # the fin rate itself, so the two factors of the design law can be read apart
            rates.append(float(np.max(np.abs(np.diff(fl["delta"]))) / (fl["t"][1] - fl["t"][0])))
    if not peaks:
        return None
    return dict(n=len(peaks), peak_median=float(np.median(peaks)),
                peak_lo=float(np.min(peaks)), peak_hi=float(np.max(peaks)),
                fin_rate_median_dps=float(np.degrees(np.median(rates))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    ML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "ml")
    ladder = json.load(open(os.path.join(ML, "dphi_ladder.json"), encoding="utf-8"))
    lo = max(r["peak"] for r in ladder if not r.get("converts"))
    hi = min(r["peak"] for r in ladder if r.get("converts"))
    print("DESIGN-LAW PREDICTION -- does the LEVER-ARM ladder's bracket predict the AIRFRAME axis?")
    print(f"  ladder bracket (from dphi_ladder.json, swept in r): [{lo:.1f}, {hi:.1f}] rad/s\n")

    conv = {}
    fp = os.path.join(ML, "airframe_class_sweep.json")
    if os.path.exists(fp):
        for r in json.load(open(fp, encoding="utf-8"))["rows"]:
            conv[r["airframe"]] = r.get("adv_median") is not None

    rows, verdicts = [], []
    print(f"  {'airframe':<12}{'peak |dphi/dt|':>26}{'fin rate':>12}{'converts':>10}{'predicted':>11}")
    for name, af in AIRFRAMES:
        r = peak_dphi(af, args.seeds)
        if r is None:
            continue
        c = conv.get(name)
        # THREE-VALUED on purpose. Rounding "inside the bracket" to either side is how a design law
        # gets confirmed by a test that could not have refuted it.
        pred = True if r["peak_median"] >= hi else (False if r["peak_median"] <= lo else None)
        agree = None if (c is None or pred is None) else (pred == c)
        r.update(airframe=name, converts=c, predicted=pred, agrees=agree)
        rows.append(r)
        verdicts.append(agree)
        print(f"  {name:<12}{r['peak_median']:>10.1f} [{r['peak_lo']:.0f},{r['peak_hi']:.0f}]"
              f"{r['fin_rate_median_dps']:>10.0f}d/s{str(c):>10}"
              f"{('converts' if pred else 'no cue') if pred is not None else 'IN BRACKET':>11}")

    decided = [v for v in verdicts if v is not None]
    holds = bool(decided) and all(decided)
    inside = sum(1 for r in rows if r["predicted"] is None)
    print(f"\n  PREDICTION HOLDS ON EVERY DECIDED AIRFRAME: {holds}"
          f"   ({len(decided)} decided, {inside} inside the bracket)")
    if not holds:
        for r in rows:
            if r["agrees"] is False:
                print(f"    MISS: {r['airframe']} peak {r['peak_median']:.1f} predicts "
                      f"{'convert' if r['predicted'] else 'no cue'}, actually "
                      f"{'converts' if r['converts'] else 'no cue'}")

    out = dict(ladder_lo=lo, ladder_hi=hi, seeds=args.seeds, quiet_snr_db=QUIET_SNR_DB,
               prediction_holds=holds, n_decided=len(decided), n_in_bracket=inside, rows=rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
