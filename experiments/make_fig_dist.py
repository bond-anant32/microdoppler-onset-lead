"""experiments/make_fig_dist.py -- per-trajectory advantages under the three reported rules.

The race figure shows one trajectory. This one shows the population the result is claimed over:
every trajectory median, under the shared threshold rule and at each matched recorded false-alarm
rate, against a zero line. It reads the stored advantage arrays and runs no simulation.

    python experiments/make_fig_dist.py

Writes study/submission/figures/fig_dist.pdf and runs/ml/fig_dist_caption.json, the latter so the
caption's numbers are asserted against the arrays rather than typed.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ML = os.path.join(ROOT, "runs", "ml")
OUT = os.path.join(ROOT, "study", "submission", "figures")
os.makedirs(OUT, exist_ok=True)

MU, CU = "#1b4965", "#c1666b"
# TrueType rather than matplotlib's default Type 3, which IEEE PDF eXpress rejects.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.6,
                     "ytick.labelsize": 6.3, "axes.linewidth": 0.7,
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                     "font.family": "serif", "mathtext.fontset": "cm"})


def load(name):
    with open(os.path.join(ML, name), encoding="utf-8") as f:
        return json.load(f)


def columns():
    ss = load("sigma_sweep_derived.json")
    op = next(r for r in ss if r["sigma"] == 0.3)
    cols = [("shared rule", np.array(op["adv"], float), MU)]
    mf = load("matched_fa_race.json")
    for row in mf["rows"]:
        if row["target_per_s"] in (0.1, 0.01) and row.get("adv"):
            lab = "$10^{-1}$/s" if row["target_per_s"] == 0.1 else "$10^{-2}$/s"
            cols.append((lab, np.array(row["adv"], float), CU))
    return cols


def main():
    cols = columns()
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    rng = np.random.default_rng(20260922)
    caption = {}
    for i, (lab, a, c) in enumerate(cols):
        x = i + rng.uniform(-0.13, 0.13, a.size)
        ax.plot(x, a, "o", ms=2.6, color=c, alpha=0.75, mew=0, zorder=3)
        med = float(np.median(a))
        ax.plot([i - 0.28, i + 0.28], [med, med], "-", color="0.15", lw=1.2, zorder=4)
        ax.annotate("%+.1f" % med, (i + 0.30, med), fontsize=6.4, color="0.15",
                    va="center", ha="left", zorder=5)
        caption["col%d" % i] = {"label": lab, "n": int(a.size), "median": med,
                                "min": float(a.min()), "all_positive": bool((a > 0).all())}
    ax.axhline(0.0, color="0.35", lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols])
    ax.set_ylabel("paired advantage (ms)")
    ax.set_xlim(-0.55, len(cols) - 0.25)
    ax.margins(y=0.14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(OUT, "fig_dist.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, "fig_dist.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)

    p = os.path.join(ML, "fig_dist_caption.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(caption, f, indent=1)
    for k in sorted(caption):
        v = caption[k]
        print("%-12s n=%-3d median %+.1f  min %+.1f  all positive %s"
              % (v["label"], v["n"], v["median"], v["min"], v["all_positive"]))
    print("wrote fig_dist + %s" % os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
