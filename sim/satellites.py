"""
sim/satellites.py - regional space-based IR sensing layer.

A down-looking LEO sensor sees a hot HGV or sea-skimmer against the cold earth with no
target-horizon limit, covering the over-ocean and over-horizon legs that ground radar cannot see.

  * Walker-Delta LEO constellation (12/2/1 by default, 18 satellites in 3 planes from
    constellation_for_corridor), H = 1000 km, near-polar, circular-Kepler propagation.
  * Down-looking passive MWIR staring FPA giving angle-only (az, el) measurements, sigma ~175 urad.
  * Signal is radiometric (sim/ir_detection.py): aerothermal skin heating
    (T_stag ~ T_amb*(1+0.2 M^2)) and boost plume, independent of RCS. A Mach-8 HGV is several
    thousand K hot; a subsonic body is cold or marginal. Tasked on hgv and supersonic_cruise by
    default; a boosting target is always observed.
  * Inverted horizon test: a down-looker sees a target unless the earth occults the line of sight.
  * Records carry z = [az, el] from the satellite's time-varying ECEF position with a 2-entry
    R_diag and feed IMM_EKF.update_angle.

The constellation is propagated in the target ECEF frame, treated as quasi-inertial over the ~200 s
flight (earth rotation of ~0.8 deg neglected), a standard short-window simplification.
"""
from __future__ import annotations
import numpy as np

from sim.ir_detection import radiant_intensity_wsr, ir_snr_db, ir_detected, fluctuate_snr_db
try:
    from trajectory_generators.geo import ecef_to_geodetic
except Exception:                                            # pragma: no cover
    ecef_to_geodetic = None

MU = 3.986004418e14          # WGS84 earth GM (m^3/s^2)
R_EARTH = 6378137.0          # WGS84 equatorial radius (m)
T_AMB = 250.0                # ambient temp at altitude (K)
SIGMA_ANGLE = 1.75e-4        # az/el 1-sigma (rad, ~175 urad staring-FPA centroid)


def walker_constellation(n_sats=12, n_planes=2, phasing=1, alt_km=1000.0, incl_deg=87.0,
                         raan_offset_deg=0.0, phase_offset_deg=0.0):
    """Walker-Delta i:T/P/F. Returns a list of sat dicts with fixed orbital elements + mean motion.
    raan_offset_deg and phase_offset_deg place the regional slice so a near-polar ground track passes
    over a threat corridor."""
    r = R_EARTH + alt_km * 1000.0
    omega = np.sqrt(MU / r ** 3)                 # mean motion (rad/s)
    incl = np.radians(incl_deg)
    per_plane = n_sats // n_planes
    ro, po = np.radians(raan_offset_deg), np.radians(phase_offset_deg)
    sats = []
    for p in range(n_planes):
        raan = 2 * np.pi * p / n_planes + ro
        for s in range(per_plane):
            phase0 = 2 * np.pi * s / per_plane + 2 * np.pi * phasing * p / n_sats + po
            sats.append(dict(id=f"leo_{p}_{s}", r=r, omega=omega, incl=incl,
                             raan=raan, phase0=phase0))
    return sats


def constellation_for_corridor(launch_lat_deg, launch_lon_deg, **kw):
    """A LEO IR slice phased so a near-polar ground track crosses the corridor: RAAN ~ launch
    longitude, and the in-plane phase set so a satellite is near the launch latitude at t=0."""
    kw.setdefault("n_sats", 18); kw.setdefault("n_planes", 3)
    return walker_constellation(raan_offset_deg=launch_lon_deg,
                                phase_offset_deg=launch_lat_deg, **kw)


def sat_position(sat, t):
    """Circular-orbit ECEF-quasi-inertial position at time t (s)."""
    u = sat["phase0"] + sat["omega"] * t         # argument of latitude
    r, i, O = sat["r"], sat["incl"], sat["raan"]
    xo, yo = r * np.cos(u), r * np.sin(u)        # in-plane
    x1, y1, z1 = xo, yo * np.cos(i), yo * np.sin(i)          # incline about x
    x = x1 * np.cos(O) - y1 * np.sin(O)                      # rotate by RAAN about z
    y = x1 * np.sin(O) + y1 * np.cos(O)
    return np.array([x, y, z1])


def has_los(sat_p, tgt_p, body_r=R_EARTH):
    """True if the earth does not occult the sat->target segment (down-looking visibility test)."""
    d = tgt_p - sat_p
    dd = float(d @ d)
    if dd < 1.0:
        return True
    s = np.clip(-float(sat_p @ d) / dd, 0.0, 1.0)           # closest-approach param on the segment
    closest = sat_p + s * d
    # Cap the occulting sphere at the target's geocentric radius. At high latitude the WGS84 surface lies
    # ~km below the equatorial R_EARTH, and a larger occulter would hide the target behind itself.
    local_body_r = min(body_r, float(np.linalg.norm(tgt_p)))
    return float(np.linalg.norm(closest)) >= local_body_r - 1.0


def _alt_coszen(p, sat_p):
    """target geodetic altitude (m) + cos(zenith) of the LOS at the target (= LOS.up_target)."""
    p = np.asarray(p, float); sat_p = np.asarray(sat_p, float)
    alt = float(ecef_to_geodetic(*p)[2]) if ecef_to_geodetic is not None else \
        float(np.linalg.norm(p) - R_EARTH)
    up = p / (np.linalg.norm(p) + 1e-9)
    los = sat_p - p; los = los / (np.linalg.norm(los) + 1e-9)
    return alt, float(np.dot(up, los))


def sat_snr_db(missile_type, flight_state, p, v, sat_p):
    """Mean in-band IR SNR (dB) for one (target, satellite) pair: radiant intensity from the target
    state (boost plume if launching, else aerothermal skin), attenuated by MWIR slant transmittance
    and 1/R^2. Radiometry and detection statistics are in sim/ir_detection.py."""
    mach = float(np.linalg.norm(v)) / 343.0
    alt, cosz = _alt_coszen(p, sat_p)
    intensity = radiant_intensity_wsr(missile_type, flight_state, mach, alt)
    R = float(np.linalg.norm(np.asarray(p, float) - np.asarray(sat_p, float)))
    return ir_snr_db(intensity, R, alt, cosz)


def _generate_ir_clutter(sat_id, sp, rng, lam):
    """Poisson(lam) IR false alarms per satellite per frame (sun glint, cloud edges, other plumes).
    Angle-only records share the schema of true detections (marginal SNR ~11-13 dB, object_id=None,
    is_clutter=True), so angle-only association faces realistic clutter. is_clutter is a truth label
    for scoring and is not a fusion input, as in sim/measurements._generate_clutter."""
    out = []
    east, north, up = _enu_basis(sp)
    cone = np.radians(55.0)                                  # earth fills a ~55-60 deg half-cone below
    for _ in range(int(rng.poisson(lam))):
        phi = rng.uniform(0, 2 * np.pi)
        th = np.arccos(1.0 - rng.random() * (1.0 - np.cos(cone)))
        los = -np.cos(th) * up + np.sin(th) * (np.cos(phi) * east + np.sin(phi) * north)
        az = np.arctan2(los[1], los[0]); el = np.arctan2(los[2], np.hypot(los[0], los[1]))
        out.append(dict(id=sat_id, ecef_m=sp, z=[float(az), float(el)],
                        R_diag=[SIGMA_ANGLE ** 2, SIGMA_ANGLE ** 2], sensor_type="sat_ir",
                        object_id=None, ir_snr_db=float(rng.uniform(11.0, 13.0)),
                        ir_pd=float("nan"), is_clutter=True))
    return out


def simulate_satellite_measurements(target_states, sats, t, rng=None, pfa=1e-6, n_frames=1,
                                    lambda_ir=3e-4, tasked_types=("hgv", "supersonic_cruise")):
    """Angle-only IR detections at time t from every satellite in `sats`: aerothermal-skin and
    boost-plume radiometry (launch detection), MWIR atmospheric extinction, probabilistic Pd (a miss
    gives the tracker a predict-only step), per-look irradiance fluctuation, and Poisson false alarms.
    Radiometry and detection are in sim/ir_detection.py (self-checks: `python sim/ir_detection.py`).

    target_states: list of (object_id, missile_type, flight_state, p_ecef(3), v_ecef(3)). A booster in
    flight_state=='boost' is always observed (launch detection) regardless of tasked_types.
    Records feed IMM_EKF.update_angle: {id, ecef_m, z=[az,el], R_diag=[s^2,s^2], sensor_type='sat_ir',
    object_id, ir_snr_db (fluctuated, reported), ir_pd, is_clutter}. az/el use the h_and_H_radar
    ECEF-delta convention so the angle-only Jacobian applies directly."""
    rng = rng or np.random.default_rng(0)
    out = []
    for sat in sats:
        sp = sat_position(sat, t)
        out.extend(_generate_ir_clutter(sat["id"], sp, rng, lambda_ir))     # false alarms per frame
        for (oid, mtype, fstate, p, v) in target_states:
            looked = (tasked_types is None or mtype in tasked_types or fstate == "boost")
            if not looked or not has_los(sp, p):
                continue
            snr_mean = sat_snr_db(mtype, fstate, p, v, sp)                  # mean SNR drives the Pd draw
            is_det, pd = ir_detected(snr_mean, rng, pfa=pfa, n_frames=n_frames)
            if not is_det:                                                  # stochastic miss (predict-only)
                continue
            snr_rep = fluctuate_snr_db(snr_mean, rng)                       # fluctuated reported value
            dx, dy, dz = (np.asarray(p, float) - sp)
            az = np.arctan2(dy, dx) + rng.normal(0, SIGMA_ANGLE)
            el = np.arctan2(dz, np.hypot(dx, dy)) + rng.normal(0, SIGMA_ANGLE)
            out.append(dict(id=sat["id"], ecef_m=sp, z=[float(az), float(el)],
                            R_diag=[SIGMA_ANGLE ** 2, SIGMA_ANGLE ** 2], sensor_type="sat_ir",
                            object_id=oid, ir_snr_db=float(snr_rep), ir_pd=float(pd), is_clutter=False))
    return out


def _enu_basis(p_ecef):
    up = p_ecef / (np.linalg.norm(p_ecef) + 1e-9)
    east = np.cross([0, 0, 1.0], up); ne = np.linalg.norm(east)
    east = east / ne if ne > 1e-9 else np.array([1.0, 0, 0])
    north = np.cross(up, east)
    return east, north, up


def ground_horizon_los(radar_ecef, tgt_p, radar_alt_m=0.0):
    """True if a ground radar has line of sight to the target (target above its local horizon).

    Same occultation test as has_los, with the sensor on the earth, so visibility is limited by the
    target horizon. The occulting-sphere radius is the local geocentric radius under the radar,
    |radar_ecef| minus radar_alt_m. An off-equator surface radar has |radar_ecef| < R_EARTH, so the
    equatorial radius would place it inside the occulter."""
    body_r = float(np.linalg.norm(radar_ecef)) - radar_alt_m
    return has_los(radar_ecef, tgt_p, body_r=body_r)
