# Micro-Doppler Maneuver-Onset Detection Against a Filter-Free Kinematic Comparator

Code and data for the IEEE Signal Processing Letters submission of the same name.

## What the paper claims

A micro-Doppler cue rendered from control-surface deflection alarms **+47.5 ms before kinematic
onset**, on 97% of realizations, with zero alarms on a cue-free channel. Over long quiescent flight
the online rule alarms at 0.015/s, against 0.44/s for the comparator under the same rule.

The comparator is Page CUSUM on the true lateral acceleration corrupted by measurement noise σ,
with no tracking filter and no process-noise model. The advantage is reported as a curve
in σ:

| σ (m/s²) | µD lead | CUSUM lead | advantage |
|---|---|---|---|
| 0     | +47.5 | +48.5 | −1.0 (comparator wins) |
| 0.003 | +47.5 | +46.5 | +1.0 (crossover) |
| 0.3   | +47.5 | +29.2 | **+17.0** |
| 3     | +47.5 | −11.1 | +57.8 |

σ is swept over three decades and enters the comparator alone, so the µD lead is invariant in it.

## Reproducing it

    python experiments/sigma_sweep.py --seeds 30 --amp-factor 0.2798 \
        --json runs/ml/sigma_sweep_derived.json     # Table I
    python experiments/make_fig_race.py             # Fig. 1 and its caption values

Every value the manuscript reports is generated from `runs/ml/*.json`.

## Where the paper's footnote points

| promised in the paper | file |
|---|---|
| per-trajectory advantages behind Table I | `runs/ml/sigma_sweep_derived.json` |
| the no-cue nulls | the `fa` column of each sweep, a detector run on a cue-free channel. `runs/ml/hgv_sweep.json` carries one per cell; `runs/ml/same_channel_null.log` is the paired-channel null |
| four-family boost-glide dwell sweep | `runs/ml/hgv_sweep.json` (`stat`, `dwell`, `det`, `fa`, `lead`) |
| the trajectory set | `runs/ml/trajs/` (31 files) |

## Layout

    experiments/            the sweeps; sigma_sweep.py produces the headline
    sim/                    airframe reduction and signal model
    filters/                tracking filters
    trajectory_generators/  one generator per target class
    runs/ml/                32 measured artifacts and 5 run logs
    runs/ml/trajs/          the flown trajectory set
    audit_dataset.py        the tier-0 physics gate
    requirements.txt        numpy, scipy, matplotlib

Run them from the repository root and the paths resolve as printed.

## Scope

One target class of five converts. The study is in simulation, at 40 dB, with one scattering centre
and no line-of-sight projection, so it assumes a favourable aspect. σ is a scalar white-noise
stand-in for kinematic-input quality and carries none of a real tracker's bandwidth, correlation or
latency. The sign survives 19 constants swept one at a time, and reverses in 5 of 23 draws when
eight axes are drawn together.

## Noise seeding

`measure()` in `experiments/multiclass_lead.py` seeds both arms from the repetition index alone
(`4000 + r`, `4991 + r`), so every trajectory is measured on the same set of noise realizations.
That is common random numbers. It tightens the paired comparison, and the per-trajectory
values entering the bootstrap and the signed-rank test are not independent draws of the noise.

Re-running the whole σ sweep with a per-trajectory offset added to both seeds, giving disjoint
streams across trajectories with pairing preserved within one, keeps the sign at every σ:

| sigma | as shipped | one independent block |
|---|---|---|
| 0.0 | -1.0 | -1.0 |
| 0.1 | **+7.0** | **+7.0** |
| 0.3 | **+17.0** | **+17.0** |
| 3.0 | +57.8 | +61.5 |

The operating point was repeated over six independent blocks. The median advantage is **+16.25 ms**,
the range **+15.00 to +17.00**, and it is positive in all six. Each block pairs on 29 or 30 of
the 30 trajectories, one moving in and out of pairing as the streams change.

The published +17.0 ms sits at the top of that range. The bootstrap interval beside it is
formed over trajectories within one block and does not carry the between-block variation.
