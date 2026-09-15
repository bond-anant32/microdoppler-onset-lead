# experiments/generate_dataset.py
#
# Per-class physical parameter draws (_sample_params) and a batch scenario generator.
# The paper's measurements use _sample_params, so every experiment flies the parameter
# distribution the generators define. The batch dataset is not used by the paper.
#
# Each generated scenario contains:
#   - force-integrated truth (gravity + drag + commanded lift/thrust) for one missile class
#     (supersonic cruise, ballistic, boost-glide HGV, MaRV or MIRV),
#   - randomized launch and target points on a threat corridor,
#   - a per-scenario radar network placed under the trajectory,
#   - multi-modal measurements: range/az/el/Doppler, RCS, SNR, quality, HRR (128) and
#     micro-Doppler (64x64), plus space-based IR angles for HGV and supersonic cruise,
#   - a per-object IMM-EKF estimate with NIS.
#
# Output per scenario in runs/dataset/scenario_XXXX/:
#   measurements.csv  one row per (time, object, sensor) detection with truth, IMM estimate, NIS,
#                     measurement, scalar signatures, missile_type, maneuver_class and
#                     signature_id (row index into the HDF5 arrays; -1 for satellite rows).
#   signatures.h5     hrr (M,128) and micro_doppler (M,64,64), float16, indexed by signature_id.
# plus runs/dataset/index.csv summarizing all scenarios.
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
from trajectory_generators.corridors import CORRIDORS, corridors_for, _great_circle_km, random_long_corridor
from sim.measurements import simulate_multiobject_measurements, _build_ctx   # multi-object scan
from sim.signatures import control_series_from_trajectory   # continuous physical signature drivers
from sim.mirv import mirv_bus                                # multi-object MIRV (bus + RVs + decoys)
from sim.satellites import (constellation_for_corridor,      # space-IR over-horizon custody
                            simulate_satellite_measurements)
from filters.imm import IMM_EKF                              # 9-state [p, v, a] IMM tracker
from audit_dataset import (audit_trajectory, audit_class_balance,   # physical-envelope checks
                                 audit_geometry_diversity)

RANGE_BUCKETS = {"short": (150.0, 400.0), "medium": (400.0, 1200.0), "long": (1200.0, 3000.0)}
MISSILE_TYPES = ["supersonic_cruise", "ballistic", "hgv", "marv", "mirv"]
# missile_type -> CLASS_SPECS key in audit_dataset (physical-envelope checks)
SPEC_KEY = {"supersonic_cruise": "supersonic", "ballistic": "ballistic", "hgv": "hgv",
            "marv": "marv", "mirv": "mirv"}

# CSV schema: one row per (object, sensor) detection. est_* is the per-object 9-state IMM estimate
# [p, v, a], followed by the covariance trace, mode probabilities [CV, CA, CT], NIS and n_radars (the
# number of radars that detected the object this scan). object_id/object_type identify the object
# (MIRV: bus/rv0/decoy0/...; single-object classes: 'tgt'). Clutter rows have an empty object_id and
# NaN state.
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
        # Hypersonic cruise (Zircon/HACM-class scramjet). The vehicle boosts from the pad to a ~24-30 km cruise
        # band at ~Mach 5.5-7.5, then flies a bounded terminal S-weave and dives onto the target.
        return dict(cruise_alt_m=float(rng.uniform(23000, 28000)),      # peak (+weave) stays under the 32 km altitude cap
                    cruise_speed_mps=float(rng.uniform(1800, 2500)),   # ~Mach 5.3-7.4
                    weave_lat_accel=float(rng.uniform(90, 140)),       # ~9-14 g lateral (headroom under the 16 g cap)
                    weave_cycles=float(rng.uniform(3.0, 5.0)),
                    boost_time=float(rng.uniform(35, 55)),
                    boost_from_ground=True,                            # climb from the pad, terminal dive
                    duration_s=float(rng.uniform(180, 260)), dt=0.5)
    if missile_type == "ballistic":
        return dict(launch_speed=float(rng.uniform(1800, 3200)),
                    launch_elev_deg=float(rng.uniform(33, 50)),
                    beta=float(rng.uniform(8000, 18000)), dt=0.5)
    if missile_type == "marv":
        # Sub-ballistic maneuvering reentry vehicle (Pershing-II class) on a depressed arc, with a bounded
        # terminal pull-up and lateral bang-bang jink. The maneuver arms only on terminal descent below
        # pullup_trigger_alt (see profiles.marv_arc), so it does not loft the vehicle.
        return dict(launch_speed=float(rng.uniform(2100, 2450)),
                    launch_elev_deg=float(rng.uniform(28, 37)),    # depressed (low-loft) arc, apogee ~150-350 km
                    beta=float(rng.uniform(12000, 20000)),         # ballistic coefficient, kg/m^2
                    pullup_trigger_alt=float(rng.uniform(50000, 62000)),  # terminal pull-up arming altitude, m
                    pullup_g=float(rng.uniform(9, 12)),            # pull-up flattens the steep reentry
                    weave_g=float(rng.uniform(12, 17)),            # lateral bang-bang jink 12-17 g (under the 25 g cap)
                    weave_cycles=float(rng.uniform(3, 5)),         # jink cycles across the terminal ground-track band
                    maneuver_dur=float(rng.uniform(30, 50)), dt=0.5)
    # hgv (boost_glide_skip's length parameter is `duration`). aero="cavh" uses the wind-tunnel CAV-H lift/drag
    # polynomials; lift saturates at low dynamic pressure and the vehicle sinks toward its equilibrium glide
    # altitude. Draws that fail the physical-envelope checks are retried.
    # Envelope after the HTV-2/CAV-H boost-glide profile of Tracy & Wright: glide entry near 50-60 km at
    # Mach ~18-20, a descending corridor toward ~25 km over thousands of km, skip-glide oscillation, and a
    # ramping lateral S-weave with a terminal jink. The generator solves its duration from the range.
    return dict(glide_alt=float(rng.uniform(48000, 58000)),
                glide_speed=float(rng.uniform(5000, 6000)),
                n_skips=int(rng.integers(6, 11)),
                skip_amp=float(rng.uniform(4000, 8000)),
                weave_lat_accel=float(rng.uniform(45, 70)),    # ramps to ~2x at terminal, ~6-11 g (under the 15 g cap)
                weave_cycles=float(rng.uniform(4, 7)),
                duration=float(rng.uniform(300, 480)), dt=0.5, aero="cavh")


def _horizon_km(alt_km, mast_km=0.03):
    """4/3-Earth radar horizon (km) to a target at alt_km, from a mast_km site."""
    return math.sqrt(2.0 * (4.0 / 3.0) * 6371.0) * (math.sqrt(mast_km) + math.sqrt(max(alt_km, 0.0)))


def _build_objects(missile_type, corridor, rng, launch_ll, target_ll, evasive=True):
    """Return (objects, dt, meta) for one scenario.

    MIRV yields a bus, N RVs and M decoys; every other class yields a single object. Each object is a
    dict {id, type, P, V, birth_step, death_step, spec_key, cmd_lat_accel}. With evasive=False the
    maneuvering classes (marv, hgv) fly their nominal path with no scripted terminal evasion. The
    dataset uses evasive=True."""
    if missile_type == "mirv":
        objs, dt, meta = mirv_bus(launch_ll, target_ll, rng=rng)
        for o in objs:
            o["cmd_lat_accel"] = None                   # unpowered -> no anticipation cue
        return objs, dt, meta
    params = _sample_params(missile_type, rng)
    if not evasive and missile_type in ("marv", "hgv"):
        params["evasive"] = False                       # nominal path, no scripted terminal jink
    P, V, dt, meta = GENERATORS[missile_type](launch_ll, target_ll, **params)
    obj = dict(id="tgt", type=missile_type, P=P, V=V, birth_step=0, death_step=len(P) - 1,
               spec_key=SPEC_KEY[missile_type], cmd_lat_accel=meta.get("cmd_lat_accel"))
    return [obj], dt, meta


def generate_scenario(sid, missile_type, corridor, rng, out_dir):
    # Corridor endpoints and land-only sensor lay-down.
    launch_ll, target_ll = corridor.sample_endpoints(rng)
    range_km = _great_circle_km(launch_ll, target_ll)
    objects, dt, meta = _build_objects(missile_type, corridor, rng, launch_ll, target_ll)
    n_total = int(meta["n_steps"])
    if n_total < 10:
        return None

    # Physical-envelope checks on every object's alive window.
    for o in objects:
        seg = slice(o["birth_step"], o["death_step"] + 1)
        passed, checks = audit_trajectory(o["P"][seg], dt, o["spec_key"])
        if not passed:
            fails = [k for k, (ok, _) in checks.items() if not ok]
            print(f"  scenario {sid:04d} ({missile_type}/{o['id']}) failed physical-envelope checks {fails}; skipped")
            return None

    radars = corridor.sensors(rng, launch_ll, target_ll)
    if len(radars) < 2:
        return None
    rmap = {r["id"]: r for r in radars}

    # Per object: true altitudes, signature drivers (control cue), kinematic onset and one IMM_EKF, whose
    # 9-state [p, v, a] estimate, covariance trace and mode probabilities are written to the CSV.
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
        # Instantaneous maneuver state: 'maneuver' only while the vehicle pulls > 2 g lateral, so the flag is clear
        # through boost, ascent and coast and set during the terminal pull-up/weave. kin_onset marks the first
        # bend and is used for the onset lead time.
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

    # Space-based IR, tasked on the classes that need over-horizon custody (HGV and supersonic cruise) and
    # phased to the corridor. It feeds each object's IMM through angle-only updates (update_angle) and keeps
    # custody through over-ocean radar gaps.
    use_sat = missile_type in ("hgv", "supersonic_cruise")
    sats = constellation_for_corridor(launch_ll[0], launch_ll[1]) if use_sat else []
    sat_rng = np.random.default_rng(sid * 100003 + 17)   # separate RNG, so satellite draws leave the
    #                                                      radar RNG stream and radar rows unchanged

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

            # Space-based IR (angle-only, tasked classes only). Runs after the radar updates, so est_* is
            # the fused state. In an over-ocean radar gap the satellite is the only sensor updating the IMM,
            # which otherwise only predicts. IR misses and false alarms come from sim.ir_detection
            # (probabilistic Pd and clutter).
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

    # DATASET_SKIP_SIGNATURES=1 skips writing signatures.h5 (HRR and micro-Doppler, ~30 MB per scenario) for
    # consumers that use kinematics only.
    if os.environ.get("DATASET_SKIP_SIGNATURES") != "1":
        with h5py.File(os.path.join(scen_dir, "signatures.h5"), "w") as h5:
            h5.create_dataset("hrr", data=np.array(hrr_list, dtype=np.float16), compression="gzip")
            h5.create_dataset("micro_doppler", data=np.array(md_list, dtype=np.float16), compression="gzip")
            h5.attrs["missile_type"] = missile_type
            h5.attrs["maneuver_class"] = maneuver_class

    # Custody handover: the dominant detecting radar over the first 40% of flight differs from the last 40%.
    T_flight = (n_total - 1) * dt
    def _dominant(lo, hi):
        c = {rid: sum(1 for tt in ts if lo <= tt < hi) for rid, ts in det_times.items()}
        c = {k: v for k, v in c.items() if v >= 3}
        return max(c, key=c.get) if c else None
    early, late = _dominant(0.0, 0.4 * T_flight), _dominant(0.6 * T_flight, T_flight + 1)
    handover = bool(early is not None and late is not None and early != late)
    custody_gap_km = round(max(0.0, range_km - 2.0 * _horizon_km(meta.get("peak_alt_m", 0) / 1000.0)), 1)

    # Onset fields refer to the primary object (single-object maneuvering classes) and do not apply to MIRV.
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


# Great-circle range limits (km) for --long, which draws random long land-to-land corridors. Ballistic apogees
# reach ~500-750 km on these corridors and ~132 km on the fixed CORRIDORS.
LONG_RANGE_KM = {"marv": (1400.0, 2200.0)}     # shorter span for MaRV, which over-energises on a long leg
LONG_RANGE_DEFAULT = (2200.0, 3400.0)


def _pick_corridor(missile_type, rng, long_mode):
    """Corridor for one attempt: a fresh random long corridor when long_mode is set, otherwise one of the
    fixed threat corridors that allow missile_type."""
    if long_mode:
        mn, mx = LONG_RANGE_KM.get(missile_type, LONG_RANGE_DEFAULT)
        return random_long_corridor(rng, [missile_type], min_km=mn, max_km=mx)
    choices = corridors_for(missile_type)
    return choices[int(rng.integers(len(choices)))]


def _gen_one(task):
    """Parallel worker: generate one scenario from a per-scenario deterministic RNG.

    The result is reproducible and independent of worker order, and differs from the sequential
    single-stream run. Module-level so it pickles under Windows 'spawn'. Returns (sid, row or None)."""
    sid, missile_type, out_dir, seed, long_mode = task
    rng = np.random.default_rng([seed, sid])
    # --long lowers the envelope-check pass rate (HGV: 32% on the fixed corridors, 20% on long ones, mostly on lateral_g
    # and altitude). With 8 retries an HGV scenario is lost 0.80^8 = 17% of the time on long corridors and
    # 0.68^8 = 4.5% on fixed ones, so long mode allows 24 retries. audit_class_balance warns if the class mix
    # ends up skewed.
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
                    help="parallel worker processes (default 1: sequential, one shared RNG stream; >1: "
                         "per-scenario deterministic seeds, which give a different, equally deterministic dataset)")
    ap.add_argument("--long", action="store_true",
                    help="draw a random long land-to-land corridor per attempt (marv 1400-2200 km, other "
                         "classes 2200-3400 km); the default is the fixed threat corridors")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Remove existing scenario_* entries and index.csv, so a loader that globs scenario_* sees only this
    # run (index.csv is the manifest).
    if os.path.isdir(args.out):
        import shutil
        for name in os.listdir(args.out):
            if name.startswith("scenario_") or name == "index.csv":
                path = os.path.join(args.out, name)
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    os.makedirs(args.out, exist_ok=True)

    # Class weights balance sequence (window) counts. Ballistic tracks are long and HGV tracks short,
    # so equal scenario counts give HGV ~19% of sequences; HGV is over-generated and ballistic
    # under-generated to compensate.
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
        # Parallel: a process pool with per-scenario deterministic seeds.
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
            # Pick a threat corridor that allows this class, retrying with a fresh corridor and endpoints
            # until the trajectory passes the physical-envelope checks and the corridor places at least two land sites
            # (--long draws a fresh random long corridor per attempt).
            row = None
            for _attempt in range(24 if args.long else 8):     # see _gen_one: --long has a lower envelope-check pass rate
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
