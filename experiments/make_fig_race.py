"""experiments/make_fig_race.py -- the two figures of the letter, end to end.

WHAT THE READER HAS TO BE ABLE TO SEE, and what the previous version failed at.

The claim is a timing difference of a few ms against a ~35 ms budget. That is three separate
questions and one panel cannot carry them:

  fig_mech   WHY a lead exists at all. The command drives the fin immediately and the trajectory
             only later, so there is an interval in which the maneuver is in the airframe and not
             yet in the state. This is smooth physics and it should look like it.
  fig_race   THAT the detector converts the interval, both arms on one axis, each normalised by its
             OWN zero-false-alarm threshold so "crosses 1.0" means "alarms" for everyone and the
             horizontal gap IS the advantage. Zoomed to where the crossings are -- the previous
             version plotted the full 45 ms, so the informative 10 ms was 20 % of the width and the
             rest was post-alarm noise that read as chaos.
  fig_adv    WHETHER it holds over the sample. All 23 paired advantages per arm, on a broken axis
             so the one catastrophic loss is VISIBLE rather than buried in prose. The cited
             literature (ru2009, zhu2012, fan2016) reports timing as a summary over trials; a
             single-trajectory trace is illustration, not evidence, so both are shown.

Everything is re-derived from the functions multiclass_lead.py uses. No number is copied from a log.
All three are authored at final printed size (single column, 3.45 in), so 1 pt here is 1 pt on paper.

    <PY> experiments/make_fig_race.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402
import matplotlib                                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                               # noqa: E402

from experiments.multiclass_lead import (                                     # noqa: E402
    drive_airframe, onset_from_achieved, class_windows, stat_cusum,
    FIN_ARM_M, KIN_NOISE,
)
from experiments.causal_dwell_test import thr_from_cruise, DT_R               # noqa: E402
from experiments.dphi_sweep import return_from_fin, stat_matched_phase        # noqa: E402

COL_W = 3.45
# Operating point of the letter: the DERIVED feasibility amplitude (aero_feasible_factor.py gives
# 0.2798 = 1/3.57), at 40 dB. Not 1/4.7 -- that is the alpha-limited alternative Sec. II states as
# a sensitivity, and it is a separate row of the amplitude sweep.
SNR, DWELL, SEED, AMP = 40.0, 0.002, 4000, 0.2798
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "study", "submission", "figures")

MU, CU, GL = "#1b4965", "#c1666b", "#5c6b73"
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3,
                     "ytick.labelsize": 6.3, "legend.fontsize": 6.2,
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7,
                     "ytick.major.width": 0.7})


def first_cross(t, s, thr):
    i = np.flatnonzero(np.asarray(s, float) > float(thr))
    return float(t[i[0]]) if i.size else None


def build():
    w = dict(class_windows("supersonic_cruise", rng=np.random.default_rng(7777))[0][0])
    w["a_cmd"] = np.asarray(w["a_cmd"], float) * AMP
    fl = drive_airframe(w["t"], w["a_cmd"], w["V"], w["alt"])
    t_on, _ = onset_from_achieved(fl["t"], fl["az"])
    tf, delta, az = fl["t"], fl["delta"], fl["az"]
    t, s = return_from_fin(tf, delta, SNR, SEED, FIN_ARM_M)
    rk = np.random.default_rng(SEED + 991)
    azr = np.abs(np.interp(t, tf, az) + rk.normal(0, KIN_NOISE, len(t)))
    # muD and CUSUM only. At a four-sample dwell a GLR arm is a monotone transform of the trailing
    # mean and produces identical alarm times, so it is not drawn as a third comparator.
    arms = [("$\\mu$D from $\\delta(t)$", stat_matched_phase(t, s, DWELL), MU, 1.9),
            ("CUSUM", stat_cusum(azr, DWELL), CU, 1.2)]
    return w, fl, t_on, t, arms


# ---------------------------------------------------------------- 1. the mechanism
def fig_mech(w, fl, t_on, ax=None):
    tf, delta, az = fl["t"], fl["delta"], fl["az"]
    tms = 1000.0 * (tf - w["t_cmd"])
    on_ms = 1000.0 * (t_on - w["t_cmd"])
    d = np.abs(delta) / (np.abs(delta).max() + 1e-30)
    a = np.abs(az) / (np.abs(az).max() + 1e-30)

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    ax.plot(tms, d, color=MU, lw=1.9, label="fin deflection $|\\delta|$", zorder=3)
    ax.plot(tms, a, color="#2a9d8f", lw=1.5, ls="--", label="achieved $|a_z|$", zorder=3)
    ax.axvline(0, color="0.6", lw=0.8, zorder=1)
    ax.axvline(on_ms, color="#2a9d8f", lw=0.9, zorder=1)
    ax.annotate("", xy=(0, 0.80), xytext=(on_ms, 0.80),
                arrowprops=dict(arrowstyle="<|-|>", color="0.25", lw=0.9,
                                shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.text(on_ms / 2, 0.845, "budget %.0f ms" % on_ms, fontsize=6.5, ha="center",
            va="bottom", color="0.25", fontweight="bold")
    ax.text(0.6, 0.03, "command", fontsize=6, color="0.5", rotation=90, va="bottom")
    ax.text(on_ms - 1.0, 0.03, "onset", fontsize=6, color="#2a9d8f", rotation=90,
            va="bottom", ha="right")
    ax.set_xlim(-4, on_ms + 12)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("time from command (ms)", labelpad=1.5)
    ax.set_ylabel("normalized", labelpad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="center right", handlelength=1.6, borderaxespad=0.4)
    ax.text(0.005, 1.0, "(a)", transform=ax.transAxes, fontsize=7.5, fontweight="bold",
            va="top")
    if own:
        fig.tight_layout(pad=0.2)
        fig.savefig(os.path.join(OUT, "fig_mech.pdf"))
        plt.close(fig)
    # THE CAPTION'S NUMBERS, EMITTED RATHER THAN READ OFF THE PLOT. A caption value that is typed
    # by hand has no artifact behind it and drifts as silently as any other second copy, so the
    # four the caption needs -- fin fraction, achieved fraction, budget, gap -- are emitted here.
    i_on = int(np.argmin(np.abs(tf - t_on)))
    return dict(on_ms=on_ms, fin_frac=float(d[i_on]), az_frac=float(a[i_on]))


# ---------------------------------------------------------------- 2. the detection
def fig_race(w, t, arms, t_on, on_ms, ax=None):
    tms = 1000.0 * (t - w["t_cmd"])
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    cross = {}
    for lab, stat, c, lw in arms:
        thr = thr_from_cruise(t, stat, t_on)
        if thr is None or not np.isfinite(thr) or thr <= 0:
            continue
        ax.plot(tms, np.asarray(stat, float) / thr, color=c, lw=lw, label=lab, zorder=3)
        tc = first_cross(t, stat, thr)
        if tc is not None:
            cross[lab] = 1000.0 * (tc - w["t_cmd"])

    ax.axhline(1.0, color="0.35", lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.axvline(0.0, color="0.6", lw=0.8, zorder=1)
    for lab, x in cross.items():
        c = MU if lab.startswith("$\\mu$D") else (CU if lab == "CUSUM" else GL)
        ax.plot(x, 1.0, "o", color=c, ms=4.2, mec="white", mew=0.7, zorder=6)

    # Annotate the gap to CUSUM specifically -- it is the headline comparator, and taking
    # min-to-max would silently span muD->GLR and skip the CUSUM crossing entirely.
    x_mu = cross.get("$\mu$D from $\delta(t)$")
    x_cu = cross.get("CUSUM")
    if x_mu is not None and x_cu is not None:
        ax.annotate("", xy=(x_mu, 0.42), xytext=(x_cu, 0.42),
                    arrowprops=dict(arrowstyle="<|-|>", color=MU, lw=1.0,
                                    shrinkA=0, shrinkB=0, mutation_scale=7))
        ax.text((x_mu + x_cu) / 2, 0.47, "%.1f ms" % (x_cu - x_mu), fontsize=7, color=MU,
                ha="center", va="bottom", fontweight="bold")
        for x, c in ((x_mu, MU), (x_cu, CU)):
            ax.plot([x, x], [0.42, 1.0], color=c, lw=0.5, ls=":", zorder=2)
    xs = sorted(v for v in cross.values() if v is not None)

    # zoom to the crossings; the onset is 30+ ms further right and would squeeze this to nothing
    hi = max(xs) + 4 if xs else 14
    ax.set_xlim(-3.0, hi)
    ax.set_ylim(0, 1.85)
    # The vline at 0 plus the axis label already say "command", so no text is needed there.
    # Park the threshold label in the one region no trace enters: mid-x, just above the line.
    ax.text(0.5 * hi, 1.055, "zero-FA threshold", fontsize=6, color="0.4", ha="center")
    ax.annotate("onset at %.0f ms" % on_ms, xy=(hi, 0.13), xytext=(hi - 0.4, 0.13),
                fontsize=6, color="#2a9d8f", ha="right", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=0.8,
                                shrinkA=2, shrinkB=0, mutation_scale=7))
    ax.set_xlabel("time from command (ms)", labelpad=1.5)
    ax.set_ylabel("statistic / own threshold", fontsize=6.5, labelpad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5, borderaxespad=0.25,
              labelspacing=0.22, bbox_to_anchor=(0.055, 1.02))
    ax.text(0.005, 1.0, "(b)", transform=ax.transAxes, fontsize=7.5, fontweight="bold",
            va="top")
    if own:
        fig.tight_layout(pad=0.2)
        fig.savefig(os.path.join(OUT, "fig_race.pdf"))
        plt.close(fig)
    return dict(gap_ms=(x_cu - x_mu) if (x_mu is not None and x_cu is not None) else None,
                mu_cross_ms=x_mu, cusum_cross_ms=x_cu)


# ---------------------------------------------------------------- 3. the evidence
def fig_adv():
    """The two leads against comparator noise. The muD line is FLAT -- that is the argument.

    THE RED CURVE IS CUSUM. The trailing mean is a different detector, 16 ms slower at the
    operating point, so the arm is taken by name from the artifact and asserted before anything is
    drawn.
    """
    jp = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "runs", "ml", "sigma_sweep_derived.json"))
    rows = [r for r in json.load(open(jp, encoding="utf-8")) if r.get("n")]
    arm = "CUSUM Page54"
    bad = [r["sigma"] for r in rows if r.get("reported_arm") != arm]
    if bad:
        raise SystemExit("fig_adv: artifact's reported_arm is not %s at sigma %s -- re-run "
                         "sigma_sweep.py" % (arm, bad))
    sig = np.array([r["sigma"] for r in rows], float)
    mu = np.array([r["muD_lead_ms"] for r in rows], float)
    kin = np.array([r["arm_lead_ms"][arm] for r in rows], float)
    # log x-axis needs a positive stand-in for sigma = 0
    x = np.where(sig <= 0, 1e-3, sig)

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    ax.semilogx(x, mu, "o-", color=MU, lw=1.9, ms=3.6, zorder=4,
                label="$\\mu$D from $\\delta(t)$")
    ax.semilogx(x, kin, "s--", color=CU, lw=1.5, ms=3.4, zorder=4,
                label="CUSUM on true $a_z + \\sigma$")
    ax.axhline(0, color="0.6", lw=0.8, ls=(0, (4, 3)), zorder=1)

    # the crossover: where the two curves meet
    below = np.where(kin < mu)[0]
    if below.size:
        xc = x[below[0]]
        ax.axvline(xc, color="0.45", lw=0.7, ls=":", zorder=2)
        ax.text(xc * 1.3, -37, "sign change", fontsize=5.9, color="0.35", ha="left")

    ax.annotate("invariant in $\\sigma$", xy=(x[3], mu[3]), xytext=(x[1] * 1.1, mu[0] - 26),
                fontsize=6.2, color=MU,
                arrowprops=dict(arrowstyle="-|>", color=MU, lw=0.7, mutation_scale=6))

    ax.set_xlabel("comparator measurement noise $\\sigma$ (m/s$^2$)", labelpad=1.5)
    ax.set_ylabel("lead before onset (ms)", labelpad=2)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
    ax.set_xticklabels(["0", "0.01", "0.1", "1"])
    # data-driven, so a change of comparator cannot silently clip a curve off the axis
    lo_y, hi_y = min(kin.min(), mu.min()), max(kin.max(), mu.max())
    pad = 0.14 * (hi_y - lo_y)
    ax.set_ylim(lo_y - pad, hi_y + 1.6 * pad)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="lower left", handlelength=1.7, borderaxespad=0.3)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.97, bottom=0.19)
    fig.savefig(os.path.join(OUT, "fig_adv.pdf"))
    fig.savefig(os.path.join(OUT, "fig_adv.png"), dpi=300)
    plt.close(fig)


def main():
    w, fl, t_on, t, arms = build()
    # The figure is authored at final printed size. Two stacked panels in a 1.58 in box cannot hold
    # the labels the axes are asked to draw: panel (a)'s x-label lands in panel (b), panel (b)'s
    # y-label runs off the left edge, and the bottom ticks fall off the bottom. Trimming margins
    # alone does not recover it, because the height itself is the binding constraint.
    #
    # Two changes fix it. The box is taller, and the saved bounding box is computed from the drawn
    # artists rather than assumed, so a label can no longer be cropped by a hard-coded margin.
    # LaTeX scales the result to \columnwidth, so the printed width is unchanged either way.
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 2.05),
                                 gridspec_kw=dict(hspace=0.55))
    mech = fig_mech(w, fl, t_on, ax=a1)
    race = fig_race(w, t, arms, t_on, mech["on_ms"], ax=a2)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.975, bottom=0.125)
    fig.savefig(os.path.join(OUT, "fig_race.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, "fig_race.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    fig_adv()

    cap = dict(seed=SEED, snr_db=SNR, amp_factor=AMP, kin_noise=KIN_NOISE, dwell=DWELL, **mech,
               **race)
    jp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "runs", "ml", "fig_race_caption.json")
    with open(jp, "w") as f:
        json.dump(cap, f, indent=1)
    print("wrote fig_race (2 panels) + fig_adv at %.2f in column width" % COL_W)
    print("caption artifact: budget %.1f ms, fin %.0f%%, |a_z| %.0f%%, gap %s ms -> %s"
          % (cap["on_ms"], 100 * cap["fin_frac"], 100 * cap["az_frac"],
             ("%.1f" % cap["gap_ms"]) if cap["gap_ms"] is not None else "n/a", jp))


if __name__ == "__main__":
    main()
