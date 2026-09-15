"""experiments/multicentre_sweep.py -- fin return plus an M-centre body interferer of random phase.

Eq. (2) writes the return as a sum over scattering centres, and the default model renders one.
scatterer_sweep.py adds a second centre and sweeps the fin's power share rho:

    s(t) = sqrt(rho) exp(j phi_fin(t)) + sqrt(1-rho) + n(t)

That interferer is real and positive, so it sits at relative phase zero on every trajectory and
noise draw. A body return from many centres at ranges incommensurate with the wavelength has
uniform phase and Rayleigh amplitude, and the interferer's phase sets where the amplitude nulls fall
relative to the fin's sweep. This script uses M centres with random phase:

    s(t) = sqrt(rho) exp(j phi_fin(t)) + sqrt(1-rho) * g_M + n(t),
    g_M  = (1/sqrt(M)) sum_{m=1..M} exp(j psi_m),   psi_m ~ U(0, 2pi) i.i.d. per realisation

M = 1 with psi = 0 is the scatterer_sweep.py model. M = 1 with random psi isolates the effect of the
phase. Large M approaches a circular complex Gaussian interferer of unit mean power, the standard
many-scatterer body return with the bulk Doppler removed.

The paper reports that under dilution the median per-trajectory advantage holds up and the
per-trajectory minimum does not. Each cell reports both statistics, so pinned and random-phase
interferers can be compared at equal rho.

    python experiments/multicentre_sweep.py --json runs/ml/multicentre_sweep.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AMP, SNR, REPS, DWELL, SIGMA, SEEDS = 0.2798, 40.0, 12, 0.002, 0.3, 30

# (rho, M, random_phase). rho = 1 reproduces the default single-centre cell (self-check).
CELLS = (
    [(1.0, 1, False)]                                        # self-check: the default model
    + [(r, 1, False) for r in (0.9, 0.5, 0.05)]              # scatterer_sweep's pinned interferer
    + [(r, 1, True) for r in (0.9, 0.5, 0.05)]               # same power, random phase
    + [(r, 8, True) for r in (0.9, 0.5, 0.05)]               # 8 centres -> near-Gaussian body return
)


def run_cell(cell):
    rho, M, rand_phase = cell
    import numpy as np
    import experiments.multiclass_lead as ml
    import experiments.dphi_sweep as dps
    from experiments.multiclass_lead import class_windows, measure
    from experiments.causal_dwell_test import DT_R, C, FC_HZ

    def return_multi(t_fin, delta, snr_db, seed, fin_arm, shift_s=0.0, cue_on=True):
        """Fin return plus an M-centre interferer; same signature as dphi_sweep.return_from_fin.

        The interferer phases are drawn from `seed`, so they are fixed within a realisation and
        vary across realisations (one body aspect per draw). Returns (t, s).
        """
        rng = np.random.default_rng(seed)
        t = np.arange(t_fin[0], t_fin[-1], DT_R)
        if cue_on:
            d = np.interp(t - shift_s, t_fin, delta, left=delta[0], right=delta[-1])
        else:
            d = np.zeros_like(t)
        lam = C / FC_HZ
        phase = 4.0 * np.pi * (fin_arm * np.sin(d)) / lam
        if rand_phase:
            psi = rng.uniform(0.0, 2.0 * np.pi, M)
        else:
            psi = np.zeros(M)
        g = np.exp(1j * psi).sum() / np.sqrt(M)               # unit mean power, random resultant
        s = np.sqrt(rho) * np.exp(1j * phase) + np.sqrt(1.0 - rho) * g
        p_n = 1.0 / (10.0 ** (snr_db / 10.0))
        s = s + np.sqrt(p_n / 2.0) * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))
        return t, s

    dps.return_from_fin = return_multi
    ml.return_from_fin = return_multi

    mus, kins, advs, dets, fas, buds = [], [], [], [], [], []
    for sd in range(SEEDS):
        try:
            wins, _ = class_windows("supersonic_cruise",
                                    rng=np.random.default_rng(90000 + sd), amp_factor=AMP)
        except Exception:                                                     # noqa: BLE001
            continue
        if not wins:
            continue
        ev_a, ev_m, ev_k = [], [], []
        for w in wins:
            w2 = dict(w)
            w2["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
            m = measure(w2, SNR, REPS, DWELL, kin_noise=SIGMA)
            if not m:
                continue
            dets.append(m["det"]); fas.append(m["fa"]); buds.append(m["budget_ms"])
            if m.get("muD") is not None:
                ev_m.append(m["muD"])
            r = (m.get("arms") or {}).get("CUSUM Page54") or {}
            if r.get("lead") is not None:
                ev_k.append(r["lead"])
            if r.get("adv") is not None:
                ev_a.append(r["adv"])
        if ev_a:
            advs.append(float(np.median(ev_a)))
        if ev_m:
            mus.append(float(np.median(ev_m)))
        if ev_k:
            kins.append(float(np.median(ev_k)))

    a = np.asarray(advs, float)
    med = lambda v: float(np.median(v)) if len(v) else None                   # noqa: E731
    return dict(rho=rho, M=M, random_phase=bool(rand_phase), n=int(a.size),
                adv_median=med(a), worst=float(a.min()) if a.size else None,
                n_pos=int((a > 0).sum()), muD_lead_ms=med(mus), cusum_lead_ms=med(kins),
                det=float(np.mean(dets)) if dets else None,
                fa=float(np.mean(fas)) if fas else None, budget_ms=med(buds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--procs", type=int, default=max(1, min(12, cpu_count() - 2)))
    args = ap.parse_args()

    print("Multi-centre sweep: fin return plus M interfering centres, pinned or random phase")
    print("  rho = fin's share of return power; M = interfering centres; phase pinned or random.")
    print("  M=1/pinned reproduces scatterer_sweep.py. rho=1 is the default model (self-check).\n")
    with Pool(args.procs, maxtasksperchild=1) as pool:
        rows = pool.map(run_cell, CELLS, chunksize=1)

    print("%6s %4s %8s %5s %11s %9s %7s %6s"
          % ("rho", "M", "phase", "n", "advantage", "worst", "pos", "det"))
    print("-" * 66)
    for r in rows:
        f = lambda v, p="%+.2f": (p % v) if v is not None else "--"           # noqa: E731
        print("%6g %4d %8s %5d %11s %9s %3d/%-3d %5.0f%%"
              % (r["rho"], r["M"], "random" if r["random_phase"] else "pinned", r["n"],
                 f(r["adv_median"]), f(r["worst"], "%+.1f"), r["n_pos"], r["n"],
                 100 * (r["det"] or 0)))

    base = next((r for r in rows if r["rho"] == 1.0), None)
    if base is not None:
        ok = base["adv_median"] is not None and abs(base["adv_median"] - 17.0) < 1e-6
        print("\nself-check: rho=1 %s the reported +17.00 ms (got %s)"
              % ("reproduces" if ok else "does not reproduce",
                 "%+.2f" % base["adv_median"] if base["adv_median"] is not None else "None"))
        if not ok:
            raise SystemExit("self-check failed -- the patched renderer is not a generalisation")

    print("\nCompare pinned and random rows at equal rho. Equal medians mean the dilution result")
    print("does not depend on the interferer's phase.")
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(dict(sigma=SIGMA, snr_db=SNR, seeds=SEEDS, reps=REPS, amp_factor=AMP,
                       cells=rows), open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
