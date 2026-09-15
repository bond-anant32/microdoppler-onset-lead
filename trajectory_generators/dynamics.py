# trajectory_generators/dynamics.py
# Force-integrated point-mass dynamics in an (approximately inertial) ECEF frame.
# Earth rotation and Coriolis are neglected, which is valid for the few-minute flight
# times modelled here; the EKF also treats ECEF as inertial. RK4 integration.
import numpy as np

MU_EARTH = 3.986004418e14   # m^3 / s^2  (gravitational parameter)
R_EARTH_M = 6371e3
RHO0 = 1.225                # kg/m^3 sea-level density
SCALE_H = 8500.0            # m  isothermal atmosphere scale height

# WGS84 ellipsoid, so altitude is measured in the WGS84 truth frame.
WGS84_A = 6378137.0
WGS84_B = WGS84_A * (1.0 - 1.0 / 298.257223563)


def wgs84_sea_level_radius(p):
    """Geocentric radius of the WGS84 ellipsoid surface at the point's geocentric
    latitude. r_ellipsoid(theta) = 1 / sqrt(cos^2 theta / a^2 + sin^2 theta / b^2),
    with sin theta = z / |p|. Altitude computed against this radius is ~0 at sea level in the
    WGS84 frame; a fixed spherical radius gives ~-2 km at mid-latitudes."""
    r = np.linalg.norm(p)
    if r < 1e-6:
        return WGS84_A
    sin_t = p[2] / r
    cos2 = max(0.0, 1.0 - sin_t * sin_t)
    return 1.0 / np.sqrt(cos2 / (WGS84_A * WGS84_A) + (sin_t * sin_t) / (WGS84_B * WGS84_B))


def gravity_accel(p):
    """Newtonian gravity toward Earth centre. p: (3,) ECEF metres."""
    r = np.linalg.norm(p)
    return -MU_EARTH * p / (r * r * r)


try:
    from .atmosphere import density as _atmo_density        # resolved once at import
except ImportError:
    _atmo_density = None


def air_density(alt_m):
    """Air density (kg/m^3) at altitude alt_m (m). Uses the layered US Standard Atmosphere table,
    or a single exponential RHO0*exp(-h/SCALE_H) if the atmosphere module cannot be imported."""
    if _atmo_density is None:
        return RHO0 * np.exp(-max(alt_m, 0.0) / SCALE_H)   # atmosphere module unavailable -> exponential
    return _atmo_density(alt_m)                             # runtime errors propagate to the caller


def drag_accel(p, v, beta):
    """
    Aerodynamic drag deceleration. beta = ballistic coefficient m/(Cd*S) [kg/m^2];
    higher beta = lower drag. a_drag = -0.5 * rho * |v| * v / beta.
    """
    if beta <= 0:
        return np.zeros(3)
    rho = air_density(altitude_of(p))
    speed = np.linalg.norm(v)
    if speed < 1e-6:
        return np.zeros(3)
    return -0.5 * rho * speed * v / beta


def altitude_of(p):
    """Altitude above the WGS84 sea-level surface (frame-consistent with the truth)."""
    return float(np.linalg.norm(p) - wgs84_sea_level_radius(p))


K_INDUCED = 0.0016   # s^2/m, lift-induced drag coefficient. Drag polar C_D = C_D0 + K*C_L^2 with lift accel
#                      ~ C_L, so induced-drag decel ~ a_lat^2. Calibrated so a sustained ~12 g maneuver bleeds
#                      ~10% of speed over ~20 s and a ~3 g weave loses almost none.
LD_MIN = 2.0         # induced-drag decel is capped at a_lat / LD_MIN. At max lift the lift/drag ratio bottoms
#                      out near 2, so the cap keeps an extreme (30 g+) command from producing a non-physical
#                      instant stop. Below the cap the quadratic polar applies.


def induced_drag_accel(v, a_lat_mag, k=K_INDUCED):
    """Lift-induced drag deceleration (m/s^2), a (3,) vector along -v, for a commanded lateral (lift)
    acceleration of magnitude a_lat_mag. Induced drag ~ C_L^2 ~ a_lat^2 (the drag polar), so maneuvering
    costs energy quadratically in a_lat, up to the cap a_lat / LD_MIN. This term adds to the parasitic
    (beta) drag from run()/drag_accel."""
    sp = float(np.linalg.norm(v))
    if sp < 1e-6 or a_lat_mag <= 0.0:
        return np.zeros(3)
    a_drag = min(k * a_lat_mag * a_lat_mag, a_lat_mag / LD_MIN)
    return -a_drag * (v / sp)


def integrate(p0, v0, accel_fn, dt, n_steps):
    """
    RK4-integrate a point mass. accel_fn(t, p, v) -> (3,) total specific force
    (acceleration); the caller composes gravity, drag and control.
    Returns (positions (n,3), velocities (n,3)).
    """
    P = np.empty((n_steps, 3), dtype=float)
    V = np.empty((n_steps, 3), dtype=float)
    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()

    def deriv(t, p, v):
        return v, accel_fn(t, p, v)

    for i in range(n_steps):
        P[i] = p
        V[i] = v
        t = i * dt
        k1p, k1v = deriv(t, p, v)
        k2p, k2v = deriv(t + 0.5 * dt, p + 0.5 * dt * k1p, v + 0.5 * dt * k1v)
        k3p, k3v = deriv(t + 0.5 * dt, p + 0.5 * dt * k2p, v + 0.5 * dt * k2v)
        k4p, k4v = deriv(t + dt, p + dt * k3p, v + dt * k3v)
        p = p + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

    return P, V
