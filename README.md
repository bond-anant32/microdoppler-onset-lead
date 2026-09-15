# Micro-Doppler Maneuver-Onset Detection Against a Filter-Free Kinematic Comparator

Simulation code and measured results for the paper of the same name.

A maneuvering vehicle deflects its control surfaces before its trajectory departs appreciably. This
code renders radar micro-Doppler from the fin deflection and races a phase-rate detector against
Page CUSUM on the true lateral acceleration plus white noise σ, with no tracking filter and no
process-noise model.

## Results

At 40 dB and σ = 0.3 m/s², the micro-Doppler detector leads CUSUM by medians of **+25.0 ms** and
**+28.0 ms** at matched measured false-alarm rates of 10⁻¹/s and 10⁻²/s, positive on every paired
trajectory (30/30 and 29/29).

Under a shared threshold rule the detector alarms a median **+47.5 ms before kinematic onset**,
against a mean command-to-onset interval of 48.9 ms, detecting on 97% of realizations with zero
alarms on a cue-free channel. σ enters only the comparator, so the micro-Doppler lead is the same
at every σ:

| σ (m/s²) | µD lead | CUSUM lead | advantage |
|---|---|---|---|
| 0     | +47.5 | +48.5 | −1.0 |
| 0.003 | +47.5 | +46.5 | +1.0 |
| 0.3   | +47.5 | +29.2 | **+17.0** |
| 3     | +47.5 | −11.1 | +57.8 |

The sign survives 19 constants swept one at a time, positive in all 62 cells where every trajectory
pairs (+5.0 to +44.5 ms), and reverses in 5 of 23 draws when eight axes, including return dilution
and command shape, are drawn together.

Command duration sets where the cue works. The phase-rate statistic detects on at least 90% of
realizations for commands up to 0.6 s and never from 4 s. The seeker-acquisition turn of a
proportional-navigation missile, rising in 0.14 s, is detected on 97% with a +28.0 ms advantage
over CUSUM on all 30 trajectories.

## Reproducing the results

Run from the repository root. Each command writes the artifact listed beside it.

| result | artifact | command |
|---|---|---|
| matched false-alarm rates (Table 1) | `runs/ml/matched_fa_race.json` | `python experiments/matched_fa_race.py --json runs/ml/matched_fa_race.json` |
| false-alarm rates under the shared rule | `runs/ml/online_fa_rate.json` | `python experiments/online_fa_rate.py --json runs/ml/online_fa_rate.json` |
| leads against comparator noise (Table 2) | `runs/ml/sigma_sweep_derived.json` | `python experiments/sigma_sweep.py --seeds 30 --amp-factor 0.2798 --json runs/ml/sigma_sweep_derived.json` |
| Fig. 1 and its caption values | `runs/ml/fig_race_caption.json` | `python experiments/make_fig_race.py` |
| Fig. 2, the per-trajectory advantages | `runs/ml/fig_dist_caption.json` | `python experiments/make_fig_dist.py` |
| one-at-a-time sensitivity | `runs/ml/sensitivity_sweep.json` | `python experiments/sensitivity_sweep.py --json runs/ml/sensitivity_sweep.json` |
| joint draws | `runs/ml/joint_mc_sweep.json` | `python experiments/joint_mc_sweep.py --trials 60 --json runs/ml/joint_mc_sweep.json` |
| command duration | `runs/ml/shape_period.json` | `python experiments/shape_period_sweep.py --json runs/ml/shape_period.json` |
| detectors at a 15 s command | `runs/ml/phase_cue.json`, `slow_cue.json`, `slow_cue_alt.json`, `slow_cue_budget.json` | `phase_cue_detector.py`, `slow_cue_detector.py`, `slow_cue_detector_alt.py`, `slow_cue_budget.py` |
| amplitude axis, 40 dB | `runs/ml/aero_feasible_40db_derived.json` | `python experiments/aero_feasible_sweep.py --seeds 30 --reps 12 --snr 40 --json runs/ml/aero_feasible_40db_derived.json` |
| amplitude axis, 20 dB | `runs/ml/aero_feasible_20db_derived.json` | `python experiments/aero_feasible_sweep.py --seeds 30 --reps 12 --snr 20 --json runs/ml/aero_feasible_20db_derived.json` |
| feasibility factor | `runs/ml/cavh_feasibility.json` | `python experiments/cavh_feasibility_check.py --json runs/ml/cavh_feasibility.json` |
| scattering centres | `runs/ml/scatterer_sweep.json`, `multicentre_sweep.json` | `scatterer_sweep.py`, `multicentre_sweep.py` |
| trailing self-calibration | `runs/ml/causal_threshold.json` | `python experiments/causal_threshold_test.py --json runs/ml/causal_threshold.json` |
| boost-glide detector sweep | `runs/ml/hgv_sweep.json`, `hgv_sweep.log` | `python experiments/hgv_detector_sweep.py --seeds 6 --json runs/ml/hgv_sweep.json > runs/ml/hgv_sweep.log` |
| proportional-navigation engagement | `runs/ml/endgame_lead.json` | `python experiments/endgame_lead.py --json runs/ml/endgame_lead.json` |
| scoring pitfalls | `runs/ml/lead_metrics.log`, `search_window.log` | `python experiments/causal_dwell_test.py --metrics --reps 30 --snr 20 10 0 > runs/ml/lead_metrics.log` and `--window-sweep --reps 30 > runs/ml/search_window.log` |
| independent noise streams | `runs/ml/seed_independence.json` | `python experiments/seed_independence.py --json runs/ml/seed_independence.json` |
| acceleration noise from radar position | printed | `python experiments/sigma_from_radar.py` |

Every sweep also runs a no-cue null, reported in its `fa` column. The flown trajectory set is in
`runs/ml/trajs/`.

## Layout

    experiments/            the sweeps; sigma_sweep.py produces Table 2
    sim/                    airframe reduction, radar and signal model
    filters/                tracking filters
    trajectory_generators/  one generator per target class, plus the engagement model
    runs/ml/                measured artifacts and run logs
    runs/ml/trajs/          the flown trajectory set
    audit_dataset.py        physical-envelope checks on generated trajectories
    requirements.txt        numpy, scipy, matplotlib

## Simulation model

Results are simulated at a 40 dB per-sample SNR unless stated. The return has one scattering centre
on the control surface at a 0.30 m lever arm, sampled at a 2 kHz PRF, at a favourable aspect with no
line-of-sight projection. A planar pitch-channel airframe reduction integrated at 10⁻⁴ s produces
both the fin history and the achieved lateral acceleration that defines onset.

## Noise seeding

`measure()` in `experiments/multiclass_lead.py` seeds both arms from the repetition index
(`4000 + r`, `4991 + r`), so every trajectory is measured on the same 12 noise realizations. These
common random numbers tighten the paired comparison. With a per-trajectory offset added to both
seeds, which gives disjoint streams across trajectories and keeps the pairing within each, the sign
holds at every σ:

| σ | common streams | independent streams |
|---|---|---|
| 0.0 | -1.0 | -1.0 |
| 0.1 | **+7.0** | **+7.0** |
| 0.3 | **+17.0** | **+17.0** |
| 3.0 | +57.8 | +61.5 |

Over six independent blocks at σ = 0.3 the median advantage is **+16.25 ms**,
range **+15.00 to +17.00**, positive in all six.
