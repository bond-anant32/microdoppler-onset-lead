# experiments/generate_dataset.py
#
# SCOPE NOTE. This letter uses exactly one thing from this file: _sample_params, which
# supplies each class's physical parameter draw so the experiments fly the same
# distribution the generators define rather than a hand-written approximation. The batch
# dataset, the EKF shadow run and the multi-modal measurement stack described below are
# not exercised by any measurement in the manuscript.
# Batch scenario generator: produces a diverse, reproducible synthetic dataset of
# MANEUVERING missile scenarios for ML training. Each scenario:
#   - force-integrated truth (gravity + drag + commanded lift/thrust) from one of
#     several missile classes (supersonic cruise / ballistic / boost-glide HGV),
#   - randomized globally-distributed launch/target at short/medium/long range,
#   - a per-scenario radar network placed under the trajectory (so long-range
#     targets are actually observed -- not tied to the fixed DC-NYC network),
#   - full multi-modal measurements: range/az/el/Doppler + RCS + SNR + quality
#     + HRR (128) + micro-Doppler (64x64),
#   - an EKF run alongside (baseline estimate + NIS) for shadow-mode comparison.
#
# Output per scenario in runs/dataset/scenario_XXXX/:
#   measurements.csv  -- one row per (time, radar) with truth, EKF estimate, NIS,
#                        measurement, scalar signatures, missile_type, maneuver_class,
#                        and signature_id (row index into the HDF5 arrays).
#   signatures.h5     -- hrr (M,128) and micro_doppler (M,64,64), float16, indexed
#                        by signature_id.
# Plus runs/dataset/index.csv summarizing all scenarios.
import os
import sys
import csv
import math
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import h5py

from trajectory_generators.profiles import GENERATORS
from trajectory_generators.geo import ecef_to_geodetic
from trajectory_generators.corridors import CORRIDORS, corridors_for, _great_circle_km, random_long_corridor  # P1
from sim.measurements import simulate_multiobject_measurements, _build_ctx   # multi-object scan
from sim.signatures import control_series_from_trajectory   # continuous physical signature drivers
from sim.mirv import mirv_bus                                # multi-object MIRV (bus + RVs + decoys)
from sim.satellites import (constellation_for_corridor,      # space-IR over-horizon custody (roadmap #6)
                            simulate_satellite_measurements)
from filters.imm import IMM_EKF                              # R3 fix: store IMM 9-state (not CV-EKF 6)
from audit_dataset import (audit_trajectory, audit_class_balance,   # Tier-0 physics gate
                                 audit_geometry_diversity)

RANGE_BUCKETS = {"short": (150.0, 400.0), "medium": (400.0, 1200.0), "long": (1200.0, 3000.0)}
MISSILE_TYPES = ["supersonic_cruise", "ballistic", "hgv", "marv", "mirv"]
# missile_type -> the CLASS_SPECS key used by the Tier-0 audit
SPEC_KEY = {"supersonic_cruise": "supersonic", "ballistic": "ballistic", "hgv": "hgv",
            "marv": "marv", "mirv": "mirv"}

# Multi-object + IMM 9-state schema (R3 fix): one row per (object, radar) detection. est_* is the
# per-object IMM estimate [p,v,a] (9-state) + covariance trace + mode probabilities [CV,CA,CT] +
# n_radars (how many radars saw this object this scan). object_id/object_type identify the object
# (MIRV = bus/rv0/decoy0/...; single-object classes = 'tgt'). Clutter rows: object_id empty, NaN state.
CSV_HEADER = [
    "t", "object_id", "object_type",
    "px", "py", "pz", "vx", "vy", "vz",
    "est_px", "est_py", "est_pz", "est_vx", "est_vy", "est_vz", "est_ax", "est_ay", "est_az",
    "p_trace", "mode_cv", "mode_ca", "mode_ct", "nis", "n_radars",
    "radar_id", "range", "az", "el", "dop",
    "rcs_dbsm", "snr_db", "quality",
    "velocity_mach", "altitude_m", "signature_id",
    "missile_type", "maneuver_class", "flight_state", "is_clutter", "sensor_type",
]


def _sample_params(missile_type, rng):
    """Randomize per-type physical parameters for trajectory diversity."""
    if missile_type == "supersonic_cruise":
        # HYPERSONIC cruise (Zircon/HACM-class scramjet), not the old Mach-2.8/14 km sub-hypersonic body: boost
        # from the pad to a ~24-30 km cruise band at ~Mach 5.5-7.5, then a bounded terminal S-weave + dive. The
        # generator now climbs from the ground (it RISES) and dives onto the target internally.
        return dict(cruise_alt_m=float(rng.uniform(23000, 28000)),      # peak (+weave) stays under the 32 km gate
                    cruise_speed_mps=float(rng.uniform(1800, 2500)),   # ~Mach 5.3-7.4
                    weave_lat_accel=float(rng.uniform(90, 140)),       # ~9-14 g lateral (headroom under the 16 g cap)
                    weave_cycles=float(rng.uniform(3.0, 5.0)),
                    boost_time=float(rng.uniform(35, 55)),
                    boost_from_ground=True,                            # RISE from the pad + terminal dive (the real profile)
                    duration_s=float(rng.uniform(180, 260)), dt=0.5)
    if missile_type == "ballistic":
        return dict(launch_speed=float(rng.uniform(1800, 3200)),
                    launch_elev_deg=float(rng.uniform(33, 50)),
                    beta=float(rng.uniform(8000, 18000)), dt=0.5)
    if missile_type == "marv":
        # A proper sub-ballistic maneuvering arc (apogee ~40-75 km -- well below a ballistic 200+ km, so still
        # distinguishable) with a BOUNDED terminal pull-up/weave at reentry. The old "keep it depressed" workaround
        # (elev 16-28, dur 20-40) existed only because the pre-fix pull-up trigger fired at APOGEE and LOFTED the
        # vehicle to 270+ km; now that the trigger arms only on genuine terminal descent (profiles.py), the maneuver
        # no longer lofts, so a normal arc + a shorter (14-22 s) terminal jink is both realistic and gate-passing.
        # Terminal maneuver amplified to the real MaRV envelope (Pershing-II class): arm HIGHER (~40 km), a
        # 5-15 g pull-up + 3-6 lateral bang-bang jinks of 12-20 g over a longer (30-50 s) terminal window.
        return dict(launch_speed=float(rng.uniform(2100, 2450)),
                    launch_elev_deg=float(rng.uniform(28, 37)),    # DEPRESSED (low-loft) so apogee stays ~150-350 km
                    beta=float(rng.uniform(12000, 20000)),         # -- a sub-ballistic MaRV, not a lofted ICBM RV
                    pullup_trigger_alt=float(rng.uniform(50000, 62000)),  # start the terminal pull-up HIGH (room to bend)
                    pullup_g=float(rng.uniform(9, 12)),            # pull-up FLATTENS the reentry (visible bend) but not so
                    weave_g=float(rng.uniform(12, 17)),            # hard it over-flattens into dense-air drag & bleeds speed
                    weave_cycles=float(rng.uniform(3, 5)),         # lateral bang-bang jink 12-17 g (under the 25 g cap)
                    maneuver_dur=float(rng.uniform(30, 50)), dt=0.5)
    # hgv (boost_glide_skip takes `duration`, not `duration_s`). aero="cavh" grounds the glide in the
    # wind-tunnel CAV-H lift/drag polynomials -> the vehicle can only SUSTAIN a lower glide (lift saturates
    # at low dynamic pressure), so the COMMANDED altitude ceiling is lowered to ~46 km (equilibrium ~24-37 km)
    # to keep most scenarios physically valid (the Tier-0 gate still rejects the rest, which retry).
    # Real HTV-2/CAV-H boost-glide envelope (Tracy&Wright): enter ~50-60 km at Mach ~18-20, sink toward ~25 km
    # over thousands of km, with 8-14 skips and an aggressive ramping lateral S-weave + terminal jink. The
    # boost_glide_skip generator now flies the full launch->target range (duration solved from range) and
    # descends its glide corridor internally.
    return dict(glide_alt=float(rng.uniform(48000, 58000)),
                glide_speed=float(rng.uniform(5000, 6000)),
                n_skips=int(rng.integers(6, 11)),
                skip_amp=float(rng.uniform(4000, 8000)),
                weave_lat_accel=float(rng.uniform(45, 70)),    # ramps to ~2x terminal -> ~6-11 g (under the 15 g cap)
                weave_cycles=float(rng.uniform(4, 7)),
                duration=float(rng.uniform(300, 480)), dt=0.5, aero="cavh")


def _horizon_km(alt_km, mast_km=0.03):
    """4/3-Earth radar horizon (km) to a target at alt_km, from a mast_km site."""
    return math.sqrt(2.0 * (4.0 / 3.0) * 6371.0) * (math.sqrt(mast_km) + math.sqrt(max(alt_km, 0.0)))


def _build_objects(missile_type, corridor, rng, launch_ll, target_ll, evasive=True):
    """Return (objects, dt, meta). MIRV -> bus + N RVs + M decoys (multi-object); every other
    class -> a single object. Each object: {id,type,P,V,birth_step,death_step,spec_key,cmd_lat_accel}.
    evasive=False -> the maneuvering classes fly their NOMINAL (predictable) path with NO scripted terminal
    evasion; the evasion is then a CLOSED-LOOP reaction to the interceptor (sim/engagement_sim). The DATASET
    keeps evasive=True (scripted maneuvers for ML training + class discrimination)."""
    if missile_type == "mirv":
        objs, dt, meta = mirv_bus(launch_ll, target_ll, rng=rng)
        for o in objs:
            o["cmd_lat_accel"] = None                   # unpowered -> no anticipation cue
        return objs, dt, meta
    params = _sample_params(missile_type, rng)
    if not evasive and missile_type in ("marv", "hgv"):
        params["evasive"] = False                       # nominal predictable path -> the terminal jink becomes a
        #                                                 CLOSED-LOOP reaction to the interceptor (engagement_sim)
    P, V, dt, meta = GENERATORS[missile_type](launch_ll, target_ll, **params)
    obj = dict(id="tgt", type=missile_type, P=P, V=V, birth_step=0, death_step=len(P) - 1,
               spec_key=SPEC_KEY[missile_type], cmd_lat_accel=meta.get("cmd_lat_accel"))
    return [obj], dt, meta


def generate_scenario(sid, missile_type, corridor, rng, out_dir):
    # P1: trajectory-follows-scenario (corridor endpoints + land-only sensor lay-down).
    launch_ll, target_ll = corridor.sample_endpoints(rng)
    range_km = _great_circle_km(launch_ll, target_ll)
    objects, dt, meta = _build_objects(missile_type, corridor, rng, launch_ll, target_ll)
    n_total = int(meta["n_steps"])
    if n_total < 10:
        return None

    # Tier-0 physics gate on EVERY object's alive window.
    for o in objects:
        seg = slice(o["birth_step"], o["death_step"] + 1)
        passed, checks = audit_trajectory(o["P"][seg], dt, o["spec_key"])
        if not passed:
            fails = [k for k, (ok, _) in checks.items() if not ok]
            print(f"  scenario {sid:04d} ({missile_type}/{o['id']}) FAILED Tier-0 {fails} -- skipped")
            return None

    radars = corridor.sensors(rng, launch_ll, target_ll)
    if len(radars) < 2:
        return None
    rmap = {r["id"]: r for r in radars}

    # Per-object: true altitudes, signature drivers (control cue), kinematic onset, and ONE IMM
    # (R3 fix: the tracker is the IMM_EKF -> we persist its 9-state [p,v,a] + covariance + mode probs).
    P0_9 = np.diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2]).astype(float)
    octx = {}
    for oi, o in enumerate(objects):
        alts_o = np.array([ecef_to_geodetic(*p)[2] for p in o["P"]])
        controls_o = control_series_from_trajectory(o["P"], o["V"], dt, o["type"], alts_o,
                                                    cmd_lat_accel=o.get("cmd_lat_accel"))
        kin_lat = np.array([c.lat_accel_mps2 for c in controls_o])
        kin_onset_o = int(np.argmax(kin_lat > 2.0 * 9.80665)) if (kin_lat > 2.0 * 9.80665).any() else -1
        b = o["birth_step"]
        x0 = np.concatenate([o["P"][b], o["V"][b], np.zeros(3)])
        octx[o["id"]] = dict(alts=alts_o, controls=controls_o, kin_onset=kin_onset_o,
                             imm=IMM_EKF(x0, P0_9, seed=sid * 100 + oi))

    def _flight_state(o, i):
        c = octx[o["id"]]
        up = o["P"][i] / np.linalg.norm(o["P"][i])
        descending = float(np.dot(o["V"][i], up)) < 0.0
        if o["type"] in ("bus", "rv") or o["type"].startswith("decoy"):   # unpowered MIRV objects: no "maneuver"
            return "terminal" if (descending and c["alts"][i] < 25000.0) else "midcourse"
        # INSTANTANEOUS maneuver state (not a monotone "has-ever-maneuvered" latch): only while the vehicle is
        # actually pulling >2 g lateral, so the flag clears through pure boost/ascent/coast and lights ONLY during
        # the bounded terminal pull-up/weave pulse. kin_onset still marks the FIRST bend for onset lead-time.
        if c["kin_onset"] >= 0 and c["controls"][i].lat_accel_mps2 > 2.0 * 9.80665:
            return "maneuver"
        if descending and c["alts"][i] < 20000.0:
            return "terminal"
        return "cruise" if o["type"] == "supersonic_cruise" else "midcourse"

    maneuver_class = meta["maneuver_class"]
    # Geometry-diversity metric: median aspect over radars x sampled points (primary object).
    P0, V0 = objects[0]["P"], objects[0]["V"]
    _asps = []
    for _r in radars:
        _re = np.array(_r["ecef_m"])
        for _i in range(0, n_total, max(1, n_total // 30)):
            _los = P0[_i] - _re; _ln = np.linalg.norm(_los); _vn = np.linalg.norm(V0[_i])
            if _ln > 1.0 and _vn > 1.0:
                _asps.append(np.degrees(np.arccos(np.clip(np.dot(_los / _ln, V0[_i] / _vn), -1.0, 1.0))))
    median_aspect_deg = float(np.median(_asps)) if _asps else 90.0

    scen_dir = os.path.join(out_dir, f"scenario_{sid:04d}")
    os.makedirs(scen_dir, exist_ok=True)
    hrr_list, md_list = [], []
    last_times, glint_states, det_times = {}, {}, {}
    n_rows = 0
    NAN = float("nan")

    # Space-IR slice: tasked ONLY on the classes that physically need over-horizon custody (HGV +
    # supersonic sea-skim), per SENSOR_ARCHITECTURE_DESIGN §1; phased to the corridor. It feeds each
    # object's IMM angle-only (update_angle) and, on over-ocean radar gaps, keeps custody alive.
    use_sat = missile_type in ("hgv", "supersonic_cruise")
    sats = constellation_for_corridor(launch_ll[0], launch_ll[1]) if use_sat else []
    sat_rng = np.random.default_rng(sid * 100003 + 17)   # DEDICATED rng: the sat never perturbs the
    #                                                      radar rng stream, so regen reproduces the
    #                                                      exact radar data + only ADDS satellite rows.

    with open(os.path.join(scen_dir, "measurements.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for i in range(n_total):
            t = i * dt
            alive = [o for o in objects if o["birth_step"] <= i <= o["death_step"]]
            if not alive:
                continue
            for o in alive:                              # predict every alive object's IMM
                octx[o["id"]]["imm"].predict(dt)

            obj_states, contexts, fstate = [], {}, {}
            for o in alive:
                fs = _flight_state(o, i)
                fstate[o["id"]] = fs
                obj_states.append((o["id"], o["P"][i], o["V"][i]))
                contexts[o["id"]] = _build_ctx(
                    {"missile_type": o["type"], "flight_state": fs, "control": octx[o["id"]]["controls"][i],
                     "swerling": 1, "detection": {"pfa": 1e-6}, "glint": {}}, o["P"][i], o["V"][i])

            meas = simulate_multiobject_measurements(
                obj_states, t, last_times, rng, contexts,
                radars=radars, glint_states=glint_states, clutter_cfg={"lambda": 0.3})
            if not meas:
                continue

            # count how many radars saw each object this scan (a track-quality feature)
            det_count = {}
            for m in meas:
                if not m.get("is_clutter"):
                    det_count[m["object_id"]] = det_count.get(m["object_id"], 0) + 1

            for m in meas:
                z = m["z"]; sig_id = len(hrr_list)
                hrr_list.append(np.asarray(m["hrr_profile"], dtype=np.float16))
                md_list.append(np.asarray(m["micro_doppler"], dtype=np.float16))
                if m.get("is_clutter"):
                    w.writerow([t, "", "clutter"] + [NAN] * 6 + [NAN] * 9
                               + [NAN, NAN, NAN, NAN, NAN, 0, m["id"], *z,
                                  m["rcs_dbsm"], m["snr_db"], m["quality"], NAN, NAN, sig_id,
                                  missile_type, maneuver_class, "clutter", 1, "radar"])
                    n_rows += 1
                    continue
                oid = m["object_id"]
                o = next(oo for oo in alive if oo["id"] == oid)
                c = octx[oid]; radar = rmap[m["id"]]
                det_times.setdefault(m["id"], []).append(t)
                nis = c["imm"].update_radar(np.array(z, dtype=float), np.diag(radar["R_diag"]), radar["ecef_m"])
                mach_i = float(np.linalg.norm(o["V"][i]) / 343.0)
                w.writerow([t, oid, o["type"], *o["P"][i], *o["V"][i], *c["imm"].x,
                            float(np.trace(c["imm"].P)), *np.round(c["imm"].mu, 4), nis, det_count.get(oid, 1),
                            m["id"], *z, m["rcs_dbsm"], m["snr_db"], m["quality"],
                            mach_i, float(c["alts"][i]), sig_id, missile_type, maneuver_class,
                            fstate[oid], 0, "radar"])
                n_rows += 1

            # --- SPACE-IR satellite (angle-only over-horizon custody; tasked classes only). Runs AFTER
            # the radar updates so est_* is the fully-fused state; on an over-ocean radar GAP the sat
            # is the only sensor and keeps the IMM alive (predict-only otherwise). ir misses/false
            # alarms come from sim.ir_detection (probabilistic Pd + clutter) -- the audited realism. ---
            if use_sat and sats:
                sat_states = [(o["id"], o["type"], fstate[o["id"]], o["P"][i], o["V"][i]) for o in alive]
                for sm in simulate_satellite_measurements(sat_states, sats, t, rng=sat_rng):
                    if sm.get("is_clutter"):
                        w.writerow([t, "", "clutter"] + [NAN] * 6 + [NAN] * 9
                                   + [NAN, NAN, NAN, NAN, NAN, 0, sm["id"], NAN, sm["z"][0], sm["z"][1], NAN,
                                      NAN, float(sm["ir_snr_db"]), NAN, NAN, NAN, -1,
                                      missile_type, maneuver_class, "clutter", 1, "sat_ir"])
                        n_rows += 1
                        continue
                    oid = sm["object_id"]; o = next(oo for oo in alive if oo["id"] == oid); c = octx[oid]
                    c["imm"].update_angle(np.asarray(sm["z"], float), np.diag(sm["R_diag"]),
                                          np.asarray(sm["ecef_m"], float))
                    mach_i = float(np.linalg.norm(o["V"][i]) / 343.0)
                    w.writerow([t, oid, o["type"], *o["P"][i], *o["V"][i], *c["imm"].x,
                                float(np.trace(c["imm"].P)), *np.round(c["imm"].mu, 4), NAN, det_count.get(oid, 0),
                                sm["id"], NAN, sm["z"][0], sm["z"][1], NAN, NAN, float(sm["ir_snr_db"]), NAN,
                                mach_i, float(c["alts"][i]), -1, missile_type, maneuver_class,
                                fstate[oid], 0, "sat_ir"])
                    n_rows += 1

    # DATASET_SKIP_SIGNATURES=1 skips the (large, ~30 MB/scenario) HRR + micro-Doppler h5 -- the residual-over-fusion
    # viz predictor is KINEMATICS-ONLY (build_windows load_hrr/load_md default off), so for that retrain the h5 is dead
    # weight that fills the disk (3 GB / 100 scenarios). The onset/discrimination CNN branches DO need it -> leave the
    # env var UNSET for those datasets.
    if os.environ.get("DATASET_SKIP_SIGNATURES") != "1":
        with h5py.File(os.path.join(scen_dir, "signatures.h5"), "w") as h5:
            h5.create_dataset("hrr", data=np.array(hrr_list, dtype=np.float16), compression="gzip")
            h5.create_dataset("micro_doppler", data=np.array(md_list, dtype=np.float16), compression="gzip")
            h5.attrs["missile_type"] = missile_type
            h5.attrs["maneuver_class"] = maneuver_class

    # Custody handover (P1): dominant detecting radar in the first third != the last third.
    T_flight = (n_total - 1) * dt
    def _dominant(lo, hi):
        c = {rid: sum(1 for tt in ts if lo <= tt < hi) for rid, ts in det_times.items()}
        c = {k: v for k, v in c.items() if v >= 3}
        return max(c, key=c.get) if c else None
    early, late = _dominant(0.0, 0.4 * T_flight), _dominant(0.6 * T_flight, T_flight + 1)
    handover = bool(early is not None and late is not None and early != late)
    custody_gap_km = round(max(0.0, range_km - 2.0 * _horizon_km(meta.get("peak_alt_m", 0) / 1000.0)), 1)

    # onset fields refer to the PRIMARY object (single-object maneuvering classes); MIRV -> N/A.
    weave_start = meta.get("weave_start_step")
    kin_onset = octx[objects[0]["id"]]["kin_onset"]
    return {
        "scenario_id": sid, "missile_type": missile_type, "maneuver_class": maneuver_class,
        "corridor": corridor.name, "range_km": round(range_km, 1),
        "launch_lat": round(launch_ll[0], 3), "launch_lon": round(launch_ll[1], 3),
        "target_lat": round(target_ll[0], 3), "target_lon": round(target_ll[1], 3),
        "n_steps": n_total, "dt": dt, "duration_s": round((n_total - 1) * dt, 1),
        "n_radars": len(radars), "n_objects": len(objects), "radar_layout": corridor.name,
        "median_aspect_deg": round(median_aspect_deg, 1), "custody_gap_km": custody_gap_km,
        "handover": handover, "transoceanic": corridor.transoceanic,
        "dispense_t": meta.get("dispense_t", -1), "n_measurements": n_rows,
        "peak_alt_km": round(meta.get("peak_alt_m", 0) / 1000, 1),
        "cmd_onset_t": round(weave_start * dt, 1) if weave_start is not None else -1,
        "kin_onset_t": round(kin_onset * dt, 1) if kin_onset >= 0 else -1,
        "anticipation_lead_s": (round((kin_onset - weave_start) * dt, 2)
                                if (kin_onset >= 0 and weave_start is not None
                                    and kin_onset >= weave_start) else -1),
    }


# DEPLOYMENT-MATCHED long corridors (--long): viz/live_scene.py flies RANDOM long land-to-land corridors, not the
# fixed short CORRIDORS -- so its ballistic lofts to ~500-750 km apogee where the old fixed-corridor dataset topped out
# at ~132 km. Training on the short corridors made the deployed track badly OOD (measured: hgv OOD 29.5 vs thr 14, ML
# residual blew up to 60-120 km). These ranges MIRROR live_scene.run_class EXACTLY so train == deploy.
LONG_RANGE_KM = {"marv": (1400.0, 2200.0)}     # MaRV over-energises on a long leg; everything else gets the full span
LONG_RANGE_DEFAULT = (2200.0, 3400.0)


def _pick_corridor(missile_type, rng, long_mode):
    """--long: a fresh RANDOM long corridor per attempt (matches live_scene's distribution). else: the canonical
    fixed threat corridors."""
    if long_mode:
        mn, mx = LONG_RANGE_KM.get(missile_type, LONG_RANGE_DEFAULT)
        return random_long_corridor(rng, [missile_type], min_km=mn, max_km=mx)
    choices = corridors_for(missile_type)
    return choices[int(rng.integers(len(choices)))]


def _gen_one(task):
    """Parallel worker: generate ONE scenario with a PER-SCENARIO deterministic RNG (order-independent and
    reproducible -> the parallel path is fully deterministic, just not identical to the sequential single-
    stream seed). Module-level so it is picklable under Windows 'spawn'. Returns (sid, row_or_None)."""
    sid, missile_type, out_dir, seed, long_mode = task
    rng = np.random.default_rng([seed, sid])
    # --long lowers the Tier-0 pass rate (measured hgv: 32% on the fixed corridors -> 20% on long ones, mostly
    # lateral_g/altitude), so 8 retries lose a whole scenario 0.80^8 = 17% of the time vs 0.68^8 = 4.5% -- and it is
    # the HGV, the very class --long exists to de-OOD, that is dropped most. More retries in long mode; each is cheap
    # next to a silently missing scenario. (audit_class_balance still WARNs if the mix ends up skewed.)
    for _ in range(24 if long_mode else 8):
        corridor = _pick_corridor(missile_type, rng, long_mode)
        row = generate_scenario(sid, missile_type, corridor, rng, out_dir)
        if row is not None:
            return sid, row
    return sid, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="number of scenarios")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="runs/dataset")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (default 1 = sequential shared-RNG, reproduces the "
                         "canonical seed sequence; >1 = per-scenario deterministic seeds, ~cores x faster, "
                         "a different-but-fully-deterministic dataset). Scenario generation is CPU-bound "
                         "and embarrassingly parallel (each scenario independent).")
    ap.add_argument("--long", action="store_true",
                    help="use RANDOM LONG corridors (marv 1400-2200, others 2200-3400 km) instead of the fixed "
                         "threat corridors -- MATCHES viz/live_scene.py's deployment distribution (high exo apogees, "
                         "long flights) so the trained ML is IN-distribution on the globe, not OOD. Use this for the "
                         "residual-over-fusion viz model (runs/ml_fus).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Start from a clean directory so stale/old-schema scenarios can't contaminate
    # a loader that globs scenario_* (index.csv is the authoritative manifest).
    if os.path.isdir(args.out):
        import shutil
        for name in os.listdir(args.out):
            if name.startswith("scenario_") or name == "index.csv":
                path = os.path.join(args.out, name)
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    os.makedirs(args.out, exist_ok=True)

    # Audit M1/H6 fix: stratified sampling weighted to balance SEQUENCE counts, not just scenario
    # counts. Ballistic tracks are long (many rows) and HGV short, so equal scenario counts gave
    # HGV only ~19% of sequences; over-generate HGV / under-generate ballistic to even the windows.
    WEIGHTS = {"supersonic_cruise": 1.0, "ballistic": 0.75, "marv": 1.1, "hgv": 1.45, "mirv": 0.7}
    tw = sum(WEIGHTS[t] for t in MISSILE_TYPES)
    plan = []
    for t in MISSILE_TYPES:
        plan += [t] * max(1, round(args.n * WEIGHTS[t] / tw))
    while len(plan) < args.n:
        plan.append(str(rng.choice(MISSILE_TYPES)))
    plan = plan[:args.n]

    index = []
    if args.jobs > 1:
        # PARALLEL: each scenario is independent -> a process pool with per-scenario deterministic seeds.
        import multiprocessing as mp
        tasks = [(s, mt, args.out, args.seed, args.long) for s, mt in enumerate(plan)]
        print(f"generating {len(tasks)} scenarios on {args.jobs} parallel workers "
              f"(per-scenario deterministic seeds)...", flush=True)
        with mp.Pool(args.jobs) as pool:
            for sid, row in pool.imap(_gen_one, tasks):      # imap -> results yielded in scenario order
                if row is None:
                    print(f"  scenario {sid:04d} skipped (degenerate after retries)", flush=True)
                    continue
                index.append(row)
                print(f"  scenario {sid:04d}: {row.get('missile_type', '?'):18s} {row['corridor']:20s} "
                      f"range={row['range_km']:7.1f}km meas={row['n_measurements']:5d} "
                      f"asp={row['median_aspect_deg']:5.1f} gap={row['custody_gap_km']:6.0f} "
                      f"ho={'Y' if row['handover'] else 'n'} lead={row['anticipation_lead_s']}", flush=True)
    else:
        sid = 0
        for missile_type in plan:
            # P1: pick a threat corridor that allows this class; retry (fresh corridor + endpoints)
            # until the trajectory passes the Tier-0 gate and the corridor placed enough land sites.
            # (--long: a fresh RANDOM long corridor per attempt, matching viz/live_scene deployment.)
            row = None
            for _attempt in range(24 if args.long else 8):     # see _gen_one: --long has a lower Tier-0 pass rate
                corridor = _pick_corridor(missile_type, rng, args.long)
                row = generate_scenario(sid, missile_type, corridor, rng, args.out)
                if row is not None:
                    break
            if row is None:
                print(f"  scenario {sid:04d} ({missile_type}) skipped (degenerate after retries)")
                sid += 1
                continue
            index.append(row)
            print(f"  scenario {sid:04d}: {missile_type:18s} {row['corridor']:20s} "
                  f"range={row['range_km']:7.1f}km meas={row['n_measurements']:5d} "
                  f"asp={row['median_aspect_deg']:5.1f} gap={row['custody_gap_km']:6.0f} "
                  f"ho={'Y' if row['handover'] else 'n'} lead={row['anticipation_lead_s']}")
            sid += 1

    if index:
        with open(os.path.join(args.out, "index.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
            w.writeheader()
            w.writerows(index)

    total_meas = sum(r["n_measurements"] for r in index)
    by_type = {t: sum(1 for r in index if r["missile_type"] == t) for t in MISSILE_TYPES}
    bal_ok, bal = audit_class_balance(by_type)
    geo_ok, geo = audit_geometry_diversity([r["median_aspect_deg"] for r in index])
    layouts = {}
    for r in index:
        layouts[r["radar_layout"]] = layouts.get(r["radar_layout"], 0) + 1
    print(f"\nGenerated {len(index)} scenarios -> {args.out}")
    print(f"  by type: {by_type}")
    print(f"  radar layouts: {layouts}")
    print(f"  [{'PASS' if bal_ok else 'WARN'}] class balance: {bal}")
    print(f"  [{'PASS' if geo_ok else 'WARN'}] geometry diversity: {geo}")
    print(f"  total measurements (= signature samples): {total_meas}")
    print(f"  index: {os.path.join(args.out, 'index.csv')}")


if __name__ == "__main__":
    main()
