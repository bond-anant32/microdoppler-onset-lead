"""
trajectory_generators/base.py - shared helpers for all trajectory generators.

Every generator returns the SAME contract: (positions (N,3) ECEF metres, dt seconds).
Velocities are recovered by finite difference via velocities_from()/make_state_fn(),
so both force-integrated and shape-based generators are consumed identically downstream.

Frame: ECEF (WGS84 via trajectory_math). SI units throughout.
Physics honesty: maneuvers are produced by BOUNDED lift/thrust added to gravity+drag,
so a = v^2/r is respected by construction; commanded specific force is clamped and eased
(raised-cosine) so there are no impulses.
"""
from __future__ import annotations
from typing import Callable, Tuple
import numpy as np

from trajectory_math import ll_to_ecef
from .dynamics import gravity_accel, drag_accel, integrate, altitude_of

G0 = 9.80665          # m/s^2
MACH = 340.29         # m/s (sea-level-ish; adequate for labelling)


# ------------------------- frames & launch ---------------------
def local_frame(p: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (up, fwd, right) unit vectors. up = radial; fwd = velocity direction;
    right = horizontal cross-track (fwd x up). Robust to near-zero speed."""
    up = p / np.linalg.norm(p)
    sp = np.linalg.norm(v)
    fwd = v / sp if sp > 1e-6 else up
    right = np.cross(fwd, up)
    nr = np.linalg.norm(right)
    right = right / nr if nr > 1e-9 else np.cross(up, np.array([1.0, 0.0, 0.0]))
    return up, fwd, right


def launch_state(launch_ll, target_ll, speed: float, elevation_deg: float,
                 alt0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Initial (position, velocity): at the launch site + alt0, heading toward the target
    along the local horizontal, pitched up by elevation_deg."""
    p0 = ll_to_ecef(*launch_ll, alt0)
    up = p0 / np.linalg.norm(p0)
    tgt = ll_to_ecef(*target_ll, 0.0)
    to_t = tgt - p0
    horiz = to_t - up * np.dot(to_t, up)
    horiz = horiz / np.linalg.norm(horiz)
    el = np.radians(elevation_deg)
    dir0 = np.cos(el) * horiz + np.sin(el) * up
    return p0, speed * dir0


# ------------------------- control helpers ---------------------
def ease(m: float) -> float:
    """Raised-cosine ease 0->1 for m in [0,1] (continuous derivative -> no impulses)."""
    m = min(1.0, max(0.0, m))
    return 0.5 * (1.0 - np.cos(np.pi * m))


def clamp(a: np.ndarray, a_max: float) -> np.ndarray:
    n = float(np.linalg.norm(a))
    return a * (a_max / n) if n > a_max else a


def make_total_accel(beta: float, control_fn: Callable) -> Callable:
    """Compose the RK4 acceleration: gravity + drag + commanded lift/thrust (control_fn).
    control_fn(t, p, v) -> commanded specific force (m/s^2), already bounded/eased."""
    def accel_fn(t, p, v):
        return gravity_accel(p) + drag_accel(p, v, beta) + control_fn(t, p, v)
    return accel_fn


# ------------------------- run & post-process ------------------
def run(p0, v0, control_fn, dt, n_steps, beta,
        stop_on_impact: bool = True, min_alt: float = 0.0) -> np.ndarray:
    """Integrate the forced point mass and return positions (N,3), truncated at first
    ground impact if stop_on_impact."""
    accel_fn = make_total_accel(beta, control_fn)
    P, _V = integrate(p0, v0, accel_fn, dt, n_steps)
    if stop_on_impact:
        alts = np.array([altitude_of(p) for p in P])
        below = np.where(alts[1:] < min_alt)[0]
        if below.size:
            return P[: below[0] + 2]     # include the impacting sample
    return P


def velocities_from(pts: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(pts, dt, axis=0)


def make_state_fn(pts: np.ndarray, dt: float) -> Callable:
    """Wrap (pts, dt) as state_fn(t)->(pos, vel), matching trajectory_math.state_fn."""
    V = velocities_from(pts, dt)
    n = len(pts)
    dur = (n - 1) * dt

    def state_fn(t):
        t = max(0.0, min(float(t), dur))
        f = t / dt
        i = min(int(f), n - 2)
        frac = f - i
        p = pts[i] + frac * (pts[i + 1] - pts[i])
        v = V[i] + frac * (V[i + 1] - V[i])
        return p, v
    return state_fn


def physics_report(pts: np.ndarray, dt: float) -> dict:
    """Diagnostics for verification: altitude band, speed range, peak lateral accel and
    the implied minimum turn radius / peak g (the a = v^2/r sanity check)."""
    V = velocities_from(pts, dt)
    A = np.gradient(V, dt, axis=0)
    speeds = np.linalg.norm(V, axis=1)
    alts = np.array([altitude_of(p) for p in pts]) / 1000.0
    # lateral acceleration = component of A perpendicular to V
    lat = []
    for v, a in zip(V, A):
        s = np.linalg.norm(v)
        if s < 1.0:
            lat.append(0.0)
            continue
        vhat = v / s
        a_lat = a - np.dot(a, vhat) * vhat
        lat.append(np.linalg.norm(a_lat))
    lat = np.array(lat)
    peak_lat = float(lat.max())
    # turn radius where lateral accel is meaningful
    mask = lat > 0.5
    min_radius = float(np.min(speeds[mask] ** 2 / lat[mask])) if mask.any() else float("inf")
    return {
        "n": len(pts),
        "duration_s": (len(pts) - 1) * dt,
        "alt_min_km": float(alts.min()),
        "alt_max_km": float(alts.max()),
        "speed_min": float(speeds.min()),
        "speed_max": float(speeds.max()),
        "peak_lateral_g": peak_lat / G0,
        "min_turn_radius_km": min_radius / 1000.0,
    }
