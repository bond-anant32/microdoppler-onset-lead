# trajectory_math.py
# Pure-Python trajectory math shared by the Blender script (main_sim.py) and all
# standalone scripts (experiments, dataset generation, ML). No bpy dependency --
# importable from any Python interpreter.
#
# COORDINATE FRAME: WGS84 ellipsoid, geodetic latitude. This matches sim/radars.py,
# interceptor.py and sim/mirv.py so all position sources share one frame. (Earlier the
# truth used a sphere with geocentric latitude, which sat ~21 km away from the WGS84
# radar frame at the same lat/lon -- self-consistent inside the EKF loop but a latent
# bug when mixing modules. Standardized to WGS84.)
import math
import numpy as np

# Spherical mean radius -- kept only for the Blender globe size and coarse altitude
# references; trajectory positions use the WGS84 conversion below.
R_EARTH_M = 6371e3

# WGS84 ellipsoid constants (identical to sim/radars.py)
WGS84_A = 6378137.0                      # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563            # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)     # first eccentricity squared

# Default scenario parameters -- keep in sync with main_sim.py
LAUNCH_LAT, LAUNCH_LON = 40.730610, -73.935242   # NYC
TARGET_LAT, TARGET_LON = 38.907192, -77.036871   # DC
LAUNCH_ALT_M = 50.0
PEAK_ALT_M = 40000.0
ANIM_SECONDS = 60.0
N_SAMPLES = 1440


def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    """Convert geodetic (WGS84) lat/lon/alt to ECEF metres. Canonical project-wide
    conversion -- interceptor.py and sim/mirv.py import this."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


# Backward-compatible alias (main_sim.py and the trajectory generators import this name).
ll_to_ecef = geodetic_to_ecef


def great_circle_points(a, b, n_points):
    """
    Slerp the great-circle arc between ECEF endpoints a, b (metres). The radius is
    interpolated linearly between |a| and |b| so the endpoints are reproduced exactly
    on the WGS84 surface (rather than snapped to a sphere of fixed radius).
    """
    ra = float(np.linalg.norm(a))
    rb = float(np.linalg.norm(b))
    a_n = a / ra
    b_n = b / rb
    cosg = max(-1.0, min(1.0, float(np.dot(a_n, b_n))))
    gamma = math.acos(cosg)
    if gamma == 0:
        return [a.copy() for _ in range(n_points)]
    sin_g = math.sin(gamma)
    pts = []
    for i in range(n_points):
        t = i / (n_points - 1)
        direction = (math.sin((1 - t) * gamma) / sin_g) * a_n \
            + (math.sin(t * gamma) / sin_g) * b_n
        radius = ra + t * (rb - ra)
        pts.append(direction * radius)
    return pts


def altitude_profile(n_points, launch_alt, peak_alt, descent_to_surface=True):
    # smooth rise and fall: raised-cosine blended with a sine envelope
    profile = []
    for i in range(n_points):
        t = i / (n_points - 1)
        h = peak_alt * math.sin(math.pi * t)  # 0 -> peak -> 0
        h = launch_alt * (1 - 0.5 * (1 - math.cos(math.pi * t))) + h * 0.5
        if not descent_to_surface:
            h = max(h, launch_alt)
        profile.append(h)
    return profile


def build_trajectory(launch_lat=LAUNCH_LAT, launch_lon=LAUNCH_LON,
                     target_lat=TARGET_LAT, target_lon=TARGET_LON,
                     n_samples=N_SAMPLES, launch_alt_m=LAUNCH_ALT_M,
                     peak_alt_m=PEAK_ALT_M, descent_to_surface=True,
                     anim_seconds=ANIM_SECONDS):
    """
    Compute the full trajectory as an (n_samples, 3) ECEF array plus the sample
    time step. Identical math to the Blender script.
    """
    p_launch = ll_to_ecef(launch_lat, launch_lon, 0.0)
    p_target = ll_to_ecef(target_lat, target_lon, 0.0)
    gc_pts = great_circle_points(p_launch, p_target, n_samples)
    alt_prof = altitude_profile(n_samples, launch_alt_m, peak_alt_m, descent_to_surface)

    world_pts_m = np.empty((n_samples, 3), dtype=float)
    for i, (p_surf, h) in enumerate(zip(gc_pts, alt_prof)):
        radial = p_surf / np.linalg.norm(p_surf)
        world_pts_m[i] = p_surf + radial * h

    dt = anim_seconds / float(n_samples - 1)
    return world_pts_m, dt


# Module-level default trajectory, built lazily on first state_fn() call.
_default_traj = {"pts": None, "dt": None, "duration": None}


def _ensure_default_trajectory():
    if _default_traj["pts"] is None:
        pts, dt = build_trajectory()
        _default_traj["pts"] = pts
        _default_traj["dt"] = dt
        _default_traj["duration"] = (len(pts) - 1) * dt
    return _default_traj


def state_fn(t):
    """
    Return (p_ecef_m, v_ecef_mps) for the default NYC->DC missile at time t seconds.
    Drop-in replacement for main_sim.blender_state_fn that works outside Blender.
    Position is linearly interpolated between trajectory samples; velocity is the
    finite-difference slope of the active segment.
    """
    d = _ensure_default_trajectory()
    pts, dt, duration = d["pts"], d["dt"], d["duration"]
    n = len(pts)

    t = max(0.0, min(float(t), duration))
    f = t / dt                      # fractional sample index
    i = min(int(f), n - 2)
    frac = f - i

    p = pts[i] + frac * (pts[i + 1] - pts[i])
    v = (pts[i + 1] - pts[i]) / dt
    return p, v


# Alias so callers written against the Blender API name also work standalone.
blender_state_fn = state_fn
