# trajectory_generators/geo.py
# Geographic helpers for building diverse, globally-distributed scenarios and for
# placing a radar network appropriate to each trajectory (so long-range targets are
# actually observed, instead of relying on the fixed DC-NYC network).
import math
import numpy as np

from trajectory_math import geodetic_to_ecef, WGS84_A, WGS84_E2, R_EARTH_M


def dest_point(lat_deg, lon_deg, bearing_deg, dist_km, R_km=6371.0):
    """Forward geodesic on a sphere: point at distance/bearing from a start point.
    Good enough for choosing scenario endpoints (the truth itself uses WGS84)."""
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    d = dist_km / R_km
    lat2 = math.asin(math.sin(lat1) * math.cos(d) +
                     math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(lat1),
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


def ecef_to_geodetic(x, y, z):
    """Closed-form Bowring conversion ECEF -> (lat_deg, lon_deg, alt_m), WGS84."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    a = WGS84_A
    e2 = WGS84_E2
    b = a * math.sqrt(1.0 - e2)
    ep2 = (a * a - b * b) / (b * b)
    th = math.atan2(z * a, p * b)
    lat = math.atan2(z + ep2 * b * math.sin(th) ** 3,
                     p - e2 * a * math.cos(th) ** 3)
    N = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), alt


def random_endpoints(rng, range_km):
    """Random launch site (avoiding poles) and a target range_km away on a random
    heading. Returns ((lat,lon)_launch, (lat,lon)_target, bearing_deg)."""
    lat0 = float(rng.uniform(-55.0, 55.0))
    lon0 = float(rng.uniform(-180.0, 180.0))
    bearing = float(rng.uniform(0.0, 360.0))
    lat1, lon1 = dest_point(lat0, lon0, bearing, range_km)
    return (lat0, lon0), (lat1, lon1), bearing


def make_radar_network(centers_ll, power_db=100.0):
    """Build a RADARS_ECEF-style list from a list of (lat,lon) ground sites."""
    radars = []
    for i, (lat, lon) in enumerate(centers_ll):
        x, y, z = geodetic_to_ecef(lat, lon, 0.0)
        radars.append({
            "id": f"r{i}",
            "lat_deg": lat, "lon_deg": lon, "alt_m": 0.0,
            "ecef_m": (float(x), float(y), float(z)),
            "update_rate_hz": 2.0,
            "sigma_range_m": 50.0,
            "sigma_az_rad": 0.0009,
            "sigma_el_rad": 0.0009,
            "sigma_doppler_mps": 3.0,
            "R_diag": [50.0 ** 2, 0.0009 ** 2, 0.0009 ** 2, 3.0 ** 2],
            "power_db": power_db,
        })
    return radars


def _bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing (deg) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x))


RADAR_LAYOUTS = ("overhead", "offset", "ahead", "behind", "ring", "side")


def radar_network_for_trajectory(P, rng=None, power_db=100.0, layout=None):
    """
    Place a radar network relative to a trajectory in a RANDOMIZED geometry, so the
    dataset spans real threat-approach geometries instead of one overhead-pass template:
      overhead - under the ground track (the old default);        side   - close broadside (~90 deg);
      offset   - standoff broadside, 50-400 km cross-track;        ahead  - ahead of the target (nose-on);
      behind   - behind the launch (tail-chase/receding);          ring   - ring at random azimuths.
    Also randomizes radar count (2-5) and per-radar noise. Each radar carries its
    'layout' so the generator can log it. Returns the radar list.
    """
    rng = rng or np.random.default_rng()
    n = len(P)
    gt = [ecef_to_geodetic(*P[i])[:2] for i in range(n)]     # (lat, lon) ground track

    def pt(f):
        return gt[min(int(f * (n - 1)), n - 1)]

    def bearing_at(f):
        i = min(int(f * (n - 1)), n - 2)
        j = min(i + max(1, n // 20), n - 1)
        return _bearing(gt[i][0], gt[i][1], gt[j][0], gt[j][1])

    if layout is None:
        layout = str(rng.choice(RADAR_LAYOUTS))
    nr = int(rng.integers(2, 6))                             # 2..5 radars
    centers = []
    if layout == "overhead":
        for k in range(nr):
            la, lo = pt((k + 0.5) / nr)
            centers.append((la, lo + 0.12 * (1 if k % 2 == 0 else -1)))
    elif layout in ("offset", "side"):
        side = 1.0 if rng.random() < 0.5 else -1.0
        dmin, dmax = (50.0, 400.0) if layout == "offset" else (20.0, 80.0)
        for k in range(nr):
            f = (k + 0.5) / nr
            la, lo = pt(f)
            centers.append(dest_point(la, lo, bearing_at(f) + 90.0 * side, float(rng.uniform(dmin, dmax))))
    elif layout == "ahead":
        la, lo = pt(0.95); b = bearing_at(0.9)
        for _ in range(nr):
            centers.append(dest_point(la, lo, b + float(rng.uniform(-40, 40)), float(rng.uniform(20, 180))))
    elif layout == "behind":
        la, lo = pt(0.05); b = bearing_at(0.1) + 180.0
        for _ in range(nr):
            centers.append(dest_point(la, lo, b + float(rng.uniform(-40, 40)), float(rng.uniform(20, 180))))
    else:  # ring
        la, lo = pt(float(rng.uniform(0.4, 0.9)))
        for _ in range(nr):
            centers.append(dest_point(la, lo, float(rng.uniform(0, 360)), float(rng.uniform(30, 150))))

    radars = make_radar_network(centers, power_db=power_db)
    for r in radars:                                         # randomize per-radar noise (M1)
        r["sigma_range_m"] = float(rng.uniform(30.0, 80.0))
        r["sigma_az_rad"] = float(rng.uniform(0.0006, 0.0015))
        r["sigma_el_rad"] = r["sigma_az_rad"]
        r["sigma_doppler_mps"] = float(rng.uniform(2.0, 5.0))
        r["R_diag"] = [r["sigma_range_m"] ** 2, r["sigma_az_rad"] ** 2,
                       r["sigma_el_rad"] ** 2, r["sigma_doppler_mps"] ** 2]
        r["layout"] = layout
    return radars
