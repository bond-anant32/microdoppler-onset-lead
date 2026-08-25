# sim/measurements.py
from __future__ import annotations
import os
import sys
import math
import importlib.util
from typing import Dict, List, Tuple, Optional
import numpy as np

# ------------- CONFIG: set your main sim path here -------------
# Example: MAIN_PATH = "/absolute/path/to/main_sim.py"
MAIN_PATH = os.environ.get("SIM_MAIN_PATH", "").strip()
# ---------------------------------------------------------------

# Helper to dynamically import a module from a file path
# Ref: importlib util recipe (Python docs)
def import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)  # [2]
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)                            # [2]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)                                           # [2]
    return mod

# Import radars (must be importable via PYTHONPATH or package)
from sim.radars import RADARS_ECEF, should_radar_emit  # [9]
from sim import signatures as _sig               # grounded scattering-center signature model
from sim.detection import detected                # Swerling Pd gate (validated vs RadarSimPy)
from trajectory_generators.geo import ecef_to_geodetic as _ecef_to_geodetic  # WGS84 altitude

PI = math.pi
def wrap_pi(a: float) -> float:
    return (a + PI) % (2.0 * PI) - PI

def clamp_el(e: float) -> float:
    half = 0.5 * PI
    if e > half: return half
    if e < -half: return -half
    return e

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v

def _aspect_angle_rad(p_ecef: np.ndarray, v_ecef: np.ndarray, r_ecef: np.ndarray) -> float:
    """
    Aspect angle between radar line-of-sight (radar->target) and target body axis proxy (velocity).
    """
    los = np.array(p_ecef) - np.array(r_ecef)
    u_los = _unit(los)
    u_vel = _unit(np.array(v_ecef))
    c = float(np.clip(np.dot(u_los, u_vel), -1.0, 1.0))
    return math.acos(c)

# === Advanced measurement surrogates (fast, deterministic, tunable) ===

# Aspect-dependent RCS surrogate calibrated to open-source physical-optics estimates.
# (nose_on_dBsm, broadside_dBsm), X-band nominal. Grounded in: DF-15 POFACETS gold anchor
# (~-17 dBsm head-on X-band; Touzopoulos/Zikidis JCM 7(1) 2017), Knott/Skolnik ch.11 lobing
# physics (cylinder broadside flash >> nose-on; engine-inlet cavity = re-entrant corner
# reflector that LIFTS supersonic-cruise nose-on), and the RAM ~-10 dB / VHF-resonance rules.
# See memory rcs-anchors.md + reference/GROUNDING_MANIFEST.md. Estimates, not measured data.
RCS_DATABASE = {
    "subsonic_cruise":   (-14.0, 5.0),    # Tomahawk/Kalibr class (stealth variants lower)
    "supersonic_cruise": (1.0, 12.0),     # BrahMos/Oniks: ramjet intake cavity lifts nose-on
    "ballistic":         (-15.0, 8.0),    # DF-15 body analog; cylindrical broadside flash
    "warhead":           (-26.0, -8.0),   # sharp-cone RV, very stealthy nose-on
    "marv":              (-20.0, -5.0),   # maneuvering RV + control-flap return
    "booster":           (-10.0, 13.0),   # cylinder "flying telephone pole"
    "hgv":               (-20.0, -2.0),   # waverider sharp edges (plasma handled separately)
    "decoy":             (-30.0, -20.0),
}


def _simulate_rcs_dbsm(missile_type: str, aspect_angle_rad: float,
                       altitude_m: float, flight_state: str, rng: Optional[np.random.Generator]) -> float:
    """
    Aspect-dependent RCS surrogate (dBsm). Interpolates nose-on -> broadside by |sin(aspect)|
    (0 at nose-on/tail-on, 1 at broadside), then applies sea-skim multipath, flight-state, and
    scintillation adjustments. Calibrated to the RCS_DATABASE open-source anchors.
    """
    nose_on, broadside = RCS_DATABASE.get(missile_type, RCS_DATABASE["supersonic_cruise"])
    aspect_gain = abs(math.sin(aspect_angle_rad))            # convention-robust (0 & pi -> nose-on)
    base = nose_on + (broadside - nose_on) * aspect_gain

    # Altitude: sea-skim multipath lift; slight high-altitude reduction
    if altitude_m < 100:
        alt_adj = 2.0
    elif altitude_m > 15000:
        alt_adj = -1.0
    else:
        alt_adj = 0.0

    # Flight-state: boost plume, exposed control surfaces during maneuver
    state_adj = {"boost": 4.0, "cruise": 0.0, "terminal": 1.5,
                 "maneuver": 2.0, "evasive": 2.0}.get(flight_state, 0.0)

    jitter = (rng.normal(0, 0.5) if rng is not None else 0.0)   # scintillation
    return float(base + alt_adj + state_adj + jitter)

def _simulate_snr_db(range_m: float, rcs_dbsm: float,
                     radar_power_db: float = 100.0,
                     snr_ref_db: float = 20.0,
                     ref_range_km: float = 100.0,
                     ref_rcs_dbsm: float = 0.0) -> float:
    """
    Calibrated single-pulse SNR (dB) from the radar range equation: R^4 falloff, linear in
    RCS, modulated by radar power. Anchored so a 0 dBsm target at ref_range_km seen by a
    nominal (power=100) radar gives snr_ref_db (~20 dB). This scale makes the Swerling Pd gate
    actually bite at long range / low RCS -- where real radars lose the track.
    """
    rk = max(range_m / 1000.0, 0.1)
    snr = (snr_ref_db + (radar_power_db - 100.0) + (rcs_dbsm - ref_rcs_dbsm)
           - 40.0 * math.log10(rk / ref_range_km))
    return float(np.clip(snr, -30.0, 60.0))


def _swerling_rcs_fluctuation(mean_dbsm: float, swerling, rng) -> float:
    """Instantaneous RCS fluctuation about the mean per Swerling model. Sw1/2 -> exponential
    (Rayleigh amplitude, 2-DoF chi-square power); Sw3/4 -> 4-DoF chi-square (dominant + small
    scatterers). Sw0/5 or None -> non-fluctuating. Works in linear m^2, returns dBsm."""
    if swerling is None or swerling in (0, 5):
        return mean_dbsm
    mean_lin = 10.0 ** (mean_dbsm / 10.0)
    s = rng.exponential(mean_lin) if swerling in (1, 2) else rng.gamma(2.0, mean_lin / 2.0)
    return 10.0 * math.log10(max(s, 1e-6))


def _glint_step(prev, sigma_rad: float, rho: float, rng) -> Tuple[float, float]:
    """One AR(1) step of time-correlated angular glint (radar phase-center wander). Returns
    (glint_az, glint_el) in radians; prev is the previous offset tuple or None."""
    p0, p1 = (0.0, 0.0) if prev is None else prev
    k = math.sqrt(max(1.0 - rho * rho, 0.0))
    return (rho * p0 + k * rng.normal(0.0, sigma_rad),
            rho * p1 + k * rng.normal(0.0, sigma_rad))


def _generate_clutter(radar: Dict, rng, lam: float,
                      range_bounds=(1000.0, 400000.0), doppler_std: float = 60.0) -> List[Dict]:
    """Poisson(lam) false alarms per scan: spurious returns at random range/az/el/Doppler in the
    FOV (birds/weather/civilian traffic). A false alarm is a MARGINAL detection, so
    it carries a realistic near-threshold SNR/quality and a noise-like signature -- NOT a sentinel
    (NaN SNR / empty arrays) that would make it trivially separable. The is_clutter=True flag is
    kept only as the ground-truth supervision label, never leaked into the input features."""
    out: List[Dict] = []
    for _ in range(int(rng.poisson(lam))):
        z = np.array([rng.uniform(*range_bounds), rng.uniform(-math.pi, math.pi),
                      rng.uniform(-0.2, math.pi / 4), rng.normal(0.0, doppler_std)], dtype=float)
        snr = float(rng.uniform(8.0, 16.0))                 # marginal: just above the Pfa gate
        # bird/weather-like signature: low structure, but a REAL array (not empty)
        hrr = np.clip(0.3 * rng.standard_normal(128) + 0.2, 0.0, None).astype(np.float32)
        md = np.clip(0.25 * rng.standard_normal((64, 64)) + 0.15, 0.0, 1.0).astype(np.float32)
        out.append({
            "id": radar["id"], "z": z, "R_diag": radar["R_diag"], "true": None,
            "rcs_dbsm": float(rng.uniform(-20.0, 5.0)), "snr_db": snr,
            "quality": float(min(max((snr - 5.0) / 20.0, 0.0), 1.0)),
            "hrr_profile": hrr.tolist(), "micro_doppler": md.tolist(), "is_clutter": True,
        })
    return out

def _simulate_hrr_profile(missile_type: str, aspect_angle_rad: float,
                          velocity_mach: float, flight_state: str,
                          n_bins: int = 128, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    prof = np.zeros(n_bins, dtype=np.float32)

    if missile_type == "booster":
        a = slice(n_bins//3, n_bins//3+10)
        b = slice(2*n_bins//3-5, 2*n_bins//3+5)
        prof[a] += 1.0
        prof[b] += 0.6
    elif missile_type == "warhead":
        c = slice(n_bins//2-2, n_bins//2+2)
        prof[c] += 1.0
    elif missile_type == "supersonic_cruise":
        # Major scattering centers: nose, body junction, intake
        nose = int(0.15*n_bins); prof[nose-2:nose+3] = 1.0
        body = int(0.4*n_bins);  prof[body-3:body+4] = 0.7
        intake = int(0.75*n_bins)
        prof[intake-2:intake+3] = 0.6 * abs(math.cos(aspect_angle_rad))
        if flight_state in ("maneuver", "evasive", "terminal"):
            # More spread under dynamic states
            span = 6
            for center in [int(0.25*n_bins), int(0.55*n_bins), int(0.8*n_bins)]:
                prof[max(center-span,0):min(center+span,n_bins)] += 0.2
    else:
        # Generic elongated body
        span = 14
        s = n_bins//2 - span//2
        prof[s:s+span] += np.hanning(span).astype(np.float32)

    noise = 0.05 if flight_state in ("maneuver","evasive","terminal") else 0.02
    prof += noise * (rng.standard_normal(n_bins).astype(np.float32))
    prof = np.clip(prof, 0.0, None)
    return prof

def _simulate_micro_doppler(missile_type: str, flight_state: str,
                            spin_hz: float = 0.0, H: int = 64, W: int = 64,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    md = np.zeros((H, W), dtype=np.float32)
    mid = H // 2
    if missile_type == "warhead":
        amp = max(2, int(H*0.08))
        for x in range(W):
            y = int(mid + amp*math.sin(2*math.pi*spin_hz*x/W))
            if 0 <= y < H: md[y, x] = 1.0
        md += 0.02*rng.standard_normal((H, W)).astype(np.float32)
    elif missile_type == "booster":
        md += 0.2*rng.random((H, W)).astype(np.float32)
    elif missile_type == "supersonic_cruise":
        if flight_state in ("maneuver","evasive","terminal"):
            # intermittent streaks indicating control activity
            for x in range(0, W, 8):
                md[:, x:x+2] = 0.5
            md += 0.05*rng.standard_normal((H, W)).astype(np.float32)
        else:
            # steady cruise: faint vibrations
            for x in range(W):
                y = int(mid + 2*math.sin(2*math.pi*50*x/W))
                if 0 <= y < H: md[y, x] = 0.3
            md += 0.02*rng.standard_normal((H, W)).astype(np.float32)
    else:
        md += 0.03*rng.standard_normal((H, W)).astype(np.float32)
    return np.clip(md, 0.0, 1.0)

# === Existing geometric measurement ===

def geom_to_measurement(
    p_ecef: Tuple[float, float, float],
    v_ecef: Tuple[float, float, float],
    r_ecef: Tuple[float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Compute ideal [range, az, el, doppler] from ECEF geometry.
    """
    px, py, pz = p_ecef
    rx, ry, rz = r_ecef
    dx, dy, dz = px - rx, py - ry, pz - rz
    R = math.sqrt(dx*dx + dy*dy + dz*dz)
    if R <= 1e-6:
        # Degenerate: return zeros (caller can skip)
        return 0.0, 0.0, 0.0, 0.0
    az = math.atan2(dy, dx)
    el = math.asin(dz / R)
    # Doppler: radial component of velocity along line-of-sight
    vx, vy, vz = v_ecef
    losx, losy, losz = dx / R, dy / R, dz / R
    dop = losx * vx + losy * vy + losz * vz
    return R, az, el, dop


def has_line_of_sight(r_ecef, p_ecef, r_earth: float = 6371000.0) -> bool:
    """True if the straight radar->target segment does not pass below the Earth (audit M3:
    horizon / LOS occlusion). Uses a spherical Earth -- the ~20 km ellipsoid difference is
    negligible for an occlusion test. A radar beyond the target's horizon can no longer see it."""
    a = np.asarray(r_ecef, dtype=float)
    b = np.asarray(p_ecef, dtype=float)
    d = b - a
    L2 = float(np.dot(d, d))
    if L2 < 1.0:
        return True
    tstar = -float(np.dot(a, d)) / L2          # param of point on segment closest to geocenter
    if tstar <= 0.0 or tstar >= 1.0:
        return True                             # closest approach is an endpoint -> not occluded
    closest = a + tstar * d
    return float(np.linalg.norm(closest)) >= r_earth - 50.0

# === Main measurement API (backwards compatible) ===

def simulate_radar_measurements(
    p_ecef: Tuple[float, float, float],
    v_ecef: Tuple[float, float, float],
    t_now: float,
    last_times: Dict[str, float],
    rng: np.random.Generator,
    meas_context: Optional[Dict] = None,
    radars: Optional[List[Dict]] = None,
    glint_states: Optional[Dict] = None,
) -> List[Dict]:
    """
    Generate noisy radar measurements for all radars that emit at t_now.

    Inputs:
      - p_ecef, v_ecef: target position/velocity in ECEF (m, m/s)
      - t_now: current sim time (s)
      - last_times: dict radar_id -> last emission time (s), updated in-place
      - rng: numpy random Generator for reproducibility
      - meas_context: optional dict to drive advanced fields, e.g.:
          {
            "missile_type": "supersonic_cruise" | "ballistic" | "hgv" | "booster" | "warhead",
            "flight_state": "boost" | "cruise" | "maneuver" | "terminal" | "evasive",
            "spin_hz": 0.0,
            "radar_power_db": 100.0
          }

    Output list items (extends previous schema with advanced fields):
      {
        "id": radar_id,
        "z": np.array([range, az, el, dop]),
        "R_diag": [σ_r², σ_az², σ_el², σ_dop²],
        "true": (R_true, az_true, el_true, dop_true),
        "rcs_dbsm": float,
        "snr_db": float,
        "quality": float in [0,1],
        "hrr_profile": List[float],   # 1D array serialized
        "micro_doppler": List[List[float]]  # 2D array serialized
      }
    """
    meas_context = meas_context or {}
    ctx = _build_ctx(meas_context, p_ecef, v_ecef)
    clutter_cfg = meas_context.get("clutter")         # dict -> Poisson false alarms
    outputs: List[Dict] = []
    radar_list = RADARS_ECEF if radars is None else radars

    for radar in radar_list:  # [9]
        rid = radar["id"]
        if not should_radar_emit(last_times.get(rid, -1e12), t_now, radar["update_rate_hz"]):  # [9]
            continue
        last_times[rid] = t_now                     # radar dwelled this scan
        if clutter_cfg is not None:                 # false alarms occur regardless of the true target
            outputs.extend(_generate_clutter(
                radar, rng, float(clutter_cfg.get("lambda", 1.0)),
                tuple(clutter_cfg.get("range_bounds", (1000.0, 400000.0))),
                float(clutter_cfg.get("doppler_std", 60.0))))
        det = _detect_target(radar, ctx["p"], ctx["v"], ctx, glint_states, rid, rng)
        if det is not None:
            det["is_clutter"] = False
            outputs.append(det)
    return outputs


def _build_ctx(meas_context: Dict, p_ecef, v_ecef) -> Dict:
    """Package the per-target measurement context (precomputes true altitude + Mach once)."""
    p = np.array(p_ecef, dtype=float); v = np.array(v_ecef, dtype=float)
    return {
        "p": p, "v": v,
        "missile_type": meas_context.get("missile_type", "supersonic_cruise"),
        "flight_state": meas_context.get("flight_state", "cruise"),
        "control": meas_context.get("control") or _sig.Control(),
        "detection": meas_context.get("detection"),
        "swerling": meas_context.get("swerling"),
        "glint": meas_context.get("glint"),
        "default_power_db": float(meas_context.get("radar_power_db", 100.0)),
        "alt_m": float(_ecef_to_geodetic(*p)[2]),     # true WGS84 altitude
        "mach": float(np.linalg.norm(v) / 343.0),
    }


def _detect_target(radar, p_ecef, v_ecef, ctx: Dict, glint_states, glint_key, rng):
    """Detect ONE target from ONE (already-emitting) radar. Returns a detection dict or None
    (missed detection / horizon-occluded). Shared by the single- and multi-object entry points."""
    R_true, az_true, el_true, dop_true = geom_to_measurement(p_ecef, v_ecef, radar["ecef_m"])
    if R_true <= 1e-6:                               # degenerate geometry
        return None
    if not has_line_of_sight(radar["ecef_m"], p_ecef):   # audit M3: Earth occludes the target
        return None
    flight_state = ctx["flight_state"]; control = ctx["control"]
    swerling_rcs = ctx["swerling"]; detection_cfg = ctx["detection"]; glint_cfg = ctx["glint"]
    power_db = float(radar.get("power_db", ctx["default_power_db"]))

    # Kinematic measurement noise
    z_range = R_true + rng.normal(0.0, radar["sigma_range_m"])
    z_az = az_true + rng.normal(0.0, radar["sigma_az_rad"])
    z_el = el_true + rng.normal(0.0, radar["sigma_el_rad"])
    z_dop = dop_true + rng.normal(0.0, radar["sigma_doppler_mps"])

    # Glint: AR(1) angular error, amplified during maneuvers; per (radar, object) state key.
    if glint_cfg is not None and glint_states is not None:
        span = float(glint_cfg.get("target_span_m", 8.0))
        amp = (float(glint_cfg.get("maneuver_gain", 4.0))
               if flight_state in ("maneuver", "evasive", "terminal") else 1.0)
        sigma_g = (span / (2.0 * max(R_true, 1.0))) * amp
        g = _glint_step(glint_states.get(glint_key), sigma_g, float(glint_cfg.get("rho", 0.85)), rng)
        glint_states[glint_key] = g
        z_az += g[0]; z_el += g[1]
    z_az = wrap_pi(z_az); z_el = clamp_el(z_el)

    # Grounded signature (sim/signatures.py) from the scattering-center model at the true altitude.
    aspect = _aspect_angle_rad(p_ecef, v_ecef, np.array(radar["ecef_m"]))
    rcs_mean = _sig.rcs_dbsm(ctx["missile_type"], aspect, ctx["alt_m"], ctx["mach"], flight_state, control, rng)

    # The Swerling Pd gate uses the MEAN SNR (the closed forms already integrate the
    # fluctuation); the instantaneous fluctuated RCS/SNR is only reported to the tracker.
    snr_mean = _simulate_snr_db(R_true, rcs_mean, radar_power_db=power_db)
    if detection_cfg is not None:
        is_det, _pd = detected(snr_mean, rng, pfa=float(detection_cfg.get("pfa", 1e-6)),
                               n_pulses=int(detection_cfg.get("n_pulses", 1)),
                               swerling=int(detection_cfg.get("swerling", swerling_rcs or 1)))
        if not is_det:
            return None
    rcs_db = (_swerling_rcs_fluctuation(rcs_mean, int(swerling_rcs), rng)
              if swerling_rcs is not None else rcs_mean)
    snr_db = _simulate_snr_db(R_true, rcs_db, radar_power_db=power_db)

    if snr_db > 10.0:
        hrr = _sig.hrr_profile(ctx["missile_type"], aspect, ctx["mach"], control, n_bins=128, rng=rng)
        md = _sig.micro_doppler(ctx["missile_type"], aspect, ctx["mach"], flight_state, control,
                                H=64, W=64, rng=rng)
    else:
        hrr = np.clip(0.1 * rng.standard_normal(128), 0.0, None).astype(np.float32)
        md = np.clip(0.1 * rng.standard_normal((64, 64)), 0.0, 1.0).astype(np.float32)
    quality = float(min(max((snr_db - 5.0) / 20.0, 0.0), 1.0))
    return {
        "id": radar["id"], "z": np.array([z_range, z_az, z_el, z_dop], dtype=float),
        "R_diag": radar["R_diag"], "true": (R_true, az_true, el_true, dop_true),
        "rcs_dbsm": float(rcs_db), "snr_db": float(snr_db), "quality": quality,
        "hrr_profile": hrr.tolist(), "micro_doppler": md.tolist(),
    }


def simulate_multiobject_measurements(obj_states, t_now, last_times, rng, contexts,
                                      radars=None, glint_states=None, clutter_cfg=None):
    """Multi-object scan (MIRV): obj_states = [(object_id, p_ecef, v_ecef), ...]; contexts maps
    object_id -> a _build_ctx() dict. Per emitting radar this scan: clutter ONCE, then a detection
    per object. Every true detection is tagged object_id; clutter is tagged object_id=None."""
    radar_list = RADARS_ECEF if radars is None else radars
    if glint_states is None:
        glint_states = {}
    outputs: List[Dict] = []
    for radar in radar_list:
        rid = radar["id"]
        if not should_radar_emit(last_times.get(rid, -1e12), t_now, radar["update_rate_hz"]):
            continue
        last_times[rid] = t_now
        if clutter_cfg is not None:
            for c in _generate_clutter(radar, rng, float(clutter_cfg.get("lambda", 1.0)),
                                       tuple(clutter_cfg.get("range_bounds", (1000.0, 400000.0))),
                                       float(clutter_cfg.get("doppler_std", 60.0))):
                c["object_id"] = None
                outputs.append(c)
        for oid, p, v in obj_states:
            det = _detect_target(radar, np.asarray(p, dtype=float), np.asarray(v, dtype=float),
                                 contexts[oid], glint_states, (rid, oid), rng)
            if det is not None:
                det["is_clutter"] = False
                det["object_id"] = oid
                outputs.append(det)
    return outputs

def main_entry():
    """
    Example entry which loads your main sim module dynamically using MAIN_PATH
    and runs a single measurement step (for demonstration).
    Set SIM_MAIN_PATH env var or edit MAIN_PATH above.
    """
    if not MAIN_PATH:
        raise RuntimeError("Set SIM_MAIN_PATH environment variable to the path of your main sim file, or edit MAIN_PATH in measurements.py")
    # Import the main module dynamically (no refactor needed)  [2][3]
    main_mod = import_from_path("sim_main_module", MAIN_PATH)
    # Expect main module to expose current truth or a getter for (p_ecef, v_ecef, t_now)
    # Adjust names to your actual main API.
    if hasattr(main_mod, "get_truth_state_ecef"):
        p_ecef, v_ecef, t_now = main_mod.get_truth_state_ecef()
    else:
        # Example fallback: look for variables
        p_ecef = getattr(main_mod, "P_ECEF", (0.0, 0.0, 0.0))
        v_ecef = getattr(main_mod, "V_ECEF", (0.0, 0.0, 0.0))
        t_now  = getattr(main_mod, "T_NOW", 0.0)
    # Seed RNG for reproducibility in experiments
    rng = np.random.default_rng(12345)
    # Track last emission times
    last_times: Dict[str, float] = {}
    # Generate measurements for this time (with a default supersonic context)
    meas = simulate_radar_measurements(
        p_ecef, v_ecef, t_now, last_times, rng,
        meas_context={"missile_type":"supersonic_cruise","flight_state":"cruise"}
    )
    # Example: print a quick summary
    for m in meas:
        rid = m["id"]
        z = m["z"]
        print(f"{rid}: range={z[0]:.1f} m, az={z[1]:.4f} rad, el={z[2]:.4f} rad, dop={z[3]:.2f} m/s | "
              f"RCS={m['rcs_dbsm']:.1f} dBsm, SNR={m['snr_db']:.1f} dB, Q={m['quality']:.2f}")

if __name__ == "__main__":
    main_entry()
