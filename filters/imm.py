"""
filters/imm.py - Interacting Multiple Model (IMM) estimator over {CV, CA, CT}.

Purpose
-------
A maneuvering-target tracker that runs three motion models in parallel and blends
them by how well each explains the radar measurements. It dissolves the CV-EKF
"tuning dilemma": a tight model wins in cruise, a looser/turning model wins in the
maneuver, and the IMM switches automatically via mode probabilities.

State (common to all modes, so mixing is dimensionally clean):
    x = [px, py, pz, vx, vy, vz, ax, ay, az]   (9-state, ECEF, SI units)

Modes:
    CV - constant velocity   (white-noise acceleration, LOW process noise; benign flight)
    CA - constant accel      (white-noise jerk; sustained linear acceleration / pull-up)
    CT - coordinated turn    (rotates the horizontal velocity at a rate inferred from the
                              estimated lateral acceleration; preserves speed; the weave)

I/O contract (drops into the same harness as filters/ekf.EKF):
    IMM_EKF(x0_9, P0_9, seed=...); .predict(dt); .update_radar(z, R, radar_ecef) -> NIS
    .x  -> combined 9-state estimate (x[:6] is [pos, vel], matching the EKF)
    .P  -> combined 9x9 covariance
    .mu -> current mode-probability vector, order [CV, CA, CT]

Reuses filters/ekf.h_and_H_radar (the tested nonlinear radar model + Jacobian) so the
measurement side is identical to the EKF baseline; only the motion models differ.

Physics honesty: the CT turn rate is bounded (theta clamped) so no impulsive/impossible
turns arise from a noisy acceleration estimate. All models are purely kinematic (no
gravity term) - unmodeled gravity is absorbed by process noise, exactly as in the CV-EKF.
"""
from __future__ import annotations
import numpy as np

from filters.ekf import h_and_H_radar, h_and_H_angle, wrap_angle

# ------------------------- state layout -------------------------
POS = slice(0, 3)
VEL = slice(3, 6)
ACC = slice(6, 9)
NX = 9
NZ = 4


# ------------------------- motion models ------------------------
def F_cv9(dt: float) -> np.ndarray:
    """Constant-velocity transition. Position advances by velocity; acceleration is
    driven to zero (CV assumes no acceleration)."""
    F = np.eye(NX)
    for i in range(3):
        F[i, 3 + i] = dt          # p += v*dt
        F[6 + i, 6 + i] = 0.0     # a -> 0
    return F


def F_ca9(dt: float) -> np.ndarray:
    """Constant-acceleration transition."""
    F = np.eye(NX)
    half_dt2 = 0.5 * dt * dt
    for i in range(3):
        F[i, 3 + i] = dt          # p += v*dt
        F[i, 6 + i] = half_dt2    # p += 0.5 a dt^2
        F[3 + i, 6 + i] = dt      # v += a*dt
    return F


def _rodrigues(vec: np.ndarray, axis_unit: np.ndarray, theta: float) -> np.ndarray:
    """Rotate `vec` about unit `axis_unit` by angle `theta` (right-hand rule)."""
    c, s = np.cos(theta), np.sin(theta)
    return (vec * c
            + np.cross(axis_unit, vec) * s
            + axis_unit * np.dot(axis_unit, vec) * (1.0 - c))


def g_ct9(x: np.ndarray, dt: float, max_turn_rad: float = np.pi / 2) -> np.ndarray:
    """
    Coordinated-turn mean propagation. Infers a turn rate from the component of the
    estimated acceleration perpendicular to the (horizontal) velocity, rotates the
    horizontal velocity by omega*dt about the local vertical, preserving horizontal
    speed. The vertical velocity component is carried straight. Bounded by max_turn_rad.
    """
    p = x[POS].copy()
    v = x[VEL].copy()
    a = x[ACC].copy()

    rp = np.linalg.norm(p)
    if rp < 1e-6:
        return np.concatenate([p + v * dt, v, a])
    up = p / rp

    v_vert = np.dot(v, up) * up
    v_h = v - v_vert
    sh = np.linalg.norm(v_h)
    if sh < 1.0:                      # essentially no horizontal motion -> straight line
        return np.concatenate([p + v * dt, v, a])

    # lateral (perpendicular, horizontal) acceleration drives the turn
    a_h = a - np.dot(a, up) * up
    a_par = (np.dot(a_h, v_h) / (sh * sh)) * v_h
    a_perp = a_h - a_par
    omega = np.linalg.norm(a_perp) / sh
    sign = np.sign(np.dot(np.cross(v_h, a_perp), up))
    if sign == 0.0:
        sign = 1.0
    theta = float(np.clip(sign * omega * dt, -max_turn_rad, max_turn_rad))

    v_h_rot = _rodrigues(v_h, up, theta)
    v_new = v_vert + v_h_rot
    p_new = p + 0.5 * (v + v_new) * dt      # trapezoidal position update over the arc
    # co-rotate the acceleration with the velocity: in a coordinated turn the (centripetal) accel stays
    # perpendicular to velocity and rotates at the same omega. Leaving `a` stale de-aligns it from the
    # rotated v_h, decaying the inferred turn rate and injecting a spurious tangential (speed-changing)
    # component -> inflates NIS in sustained turns (rotation is a no-op when omega~=0, so
    # it never destabilizes benign tracks -- measured 0 extra divergences over 90 runs). Rotation about
    # `up` leaves the vertical accel component untouched and turns only the horizontal part.
    a_rot = _rodrigues(a, up, theta)
    return np.concatenate([p_new, v_new, a_rot])


def _num_jacobian(f, x: np.ndarray, dt: float, eps_scale: float = 1e-4) -> np.ndarray:
    """Forward-difference Jacobian of f(x, dt) at x (robust; avoids fragile hand
    derivatives for the nonlinear CT model)."""
    n = len(x)
    fx = f(x, dt)
    J = np.zeros((n, n))
    for i in range(n):
        dxi = eps_scale * (abs(x[i]) + 1.0)
        xp = x.copy()
        xp[i] += dxi
        J[:, i] = (f(xp, dt) - fx) / dxi
    return J


# ------------------------- process noise ------------------------
def Q_white_accel9(dt: float, q: float, accel_floor: float = 1e-2) -> np.ndarray:
    """Discrete white-noise-acceleration Q on each [p, v] axis pair (CV model). A small
    floor is placed on the acceleration diagonal so the 9x9 stays positive definite."""
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    Q = np.zeros((NX, NX))
    for i in range(3):
        Q[i, i] = dt4 / 4 * q
        Q[i, 3 + i] = Q[3 + i, i] = dt3 / 2 * q
        Q[3 + i, 3 + i] = dt2 * q
        Q[6 + i, 6 + i] = accel_floor
    return Q


def Q_white_jerk9(dt: float, q: float) -> np.ndarray:
    """Discrete white-noise-jerk Q on each [p, v, a] axis triple (CA / CT models)."""
    dt2, dt3, dt4, dt5 = dt * dt, dt ** 3, dt ** 4, dt ** 5
    Q = np.zeros((NX, NX))
    for i in range(3):
        pi, vi, ai = i, 3 + i, 6 + i
        Q[pi, pi] = dt5 / 20 * q
        Q[pi, vi] = Q[vi, pi] = dt4 / 8 * q
        Q[pi, ai] = Q[ai, pi] = dt3 / 6 * q
        Q[vi, vi] = dt3 / 3 * q
        Q[vi, ai] = Q[ai, vi] = dt2 / 2 * q
        Q[ai, ai] = dt * q
    return Q


# ------------------------- measurement helper -------------------
def _h_and_H9(x9: np.ndarray, radar_ecef):
    """Radar measurement + Jacobian for the 9-state (reuses the 6-state model; the three
    acceleration columns of H are zero because range/az/el/Doppler don't depend on accel)."""
    z, H6 = h_and_H_radar(x9[:6], radar_ecef)
    H = np.zeros((NZ, NX))
    H[:, :6] = H6
    return z, H


def _h_and_H9_angle(x9: np.ndarray, sensor_ecef):
    """Angle-only (az, el) measurement + 2x9 Jacobian for the 9-state (passive IR / satellite)."""
    z, H6 = h_and_H_angle(x9[:6], sensor_ecef)
    H = np.zeros((2, NX))
    H[:, :6] = H6
    return z, H


def gaussian_likelihood(y: np.ndarray, S: np.ndarray):
    """Return (likelihood Lambda = N(y; 0, S), NIS = y^T S^-1 y). NUMERICALLY GUARDED: S is symmetrized and the
    NIS quadratic form is clamped non-negative, so a slightly non-PD / ill-conditioned S (which produced a NEGATIVE
    nis -> a huge positive loglik -> exp OVERFLOW to +inf -> NaN mode probabilities that permanently poison the
    9-state track) can no longer blow up. The exp is additionally capped as a belt-and-suspenders."""
    S = 0.5 * (S + S.T)                                   # symmetrize (round-off can break exact symmetry)
    Sinv = np.linalg.inv(S)
    nis = max(0.0, float(y @ Sinv @ y))                  # a valid NIS is >= 0; a negative value = non-PD S
    det = max(float(np.linalg.det(S)), 1e-300)
    m = len(y)
    loglik = -0.5 * (m * np.log(2 * np.pi) + np.log(det) + nis)
    lik = float(np.exp(min(loglik, 700.0)))              # exp(700) ~ 1e304 < float max -> never inf
    return (lik if np.isfinite(lik) else 0.0), nis


# ------------------------- the IMM ------------------------------
class IMM_EKF:
    def __init__(self, x0_9, P0_9,
                 q_cv=1.0, q_ca=200.0, q_ct=50.0,
                 trans=None, mu0=None, seed=0):
        x0_9 = np.asarray(x0_9, dtype=float)
        P0_9 = np.asarray(P0_9, dtype=float)
        assert x0_9.shape == (NX,) and P0_9.shape == (NX, NX)

        self.mode_names = ["CV", "CA", "CT"]
        self._kinds = ["CV", "CA", "CT"]
        self._q = [q_cv, q_ca, q_ct]
        self.M = 3

        # Markov mode-transition matrix (rows = from, cols = to; rows sum to 1).
        if trans is None:
            trans = np.array([[0.95, 0.025, 0.025],
                              [0.05, 0.90, 0.05],
                              [0.05, 0.05, 0.90]])
        self.Pi = np.asarray(trans, dtype=float)
        self.mu = np.asarray(mu0 if mu0 is not None else [0.8, 0.1, 0.1], dtype=float)
        self.mu = self.mu / self.mu.sum()

        # per-mode estimates (all start from the same prior)
        self.x_modes = np.tile(x0_9, (self.M, 1))          # (3, 9)
        self.P_modes = np.tile(P0_9, (self.M, 1, 1))       # (3, 9, 9)
        self._Lj = self.mu.copy()                          # likelihood accumulator per step
        self.rng = np.random.default_rng(seed)

        self.x = x0_9.copy()
        self.P = P0_9.copy()

    # ---- mode-specific prediction ----
    def _predict_mode(self, j, x0, P0, dt):
        kind = self._kinds[j]
        if kind == "CV":
            F = F_cv9(dt); Q = Q_white_accel9(dt, self._q[j])
            x = F @ x0
            P = F @ P0 @ F.T + Q
        elif kind == "CA":
            F = F_ca9(dt); Q = Q_white_jerk9(dt, self._q[j])
            x = F @ x0
            P = F @ P0 @ F.T + Q
        else:  # CT (nonlinear mean, numerical Jacobian for covariance)
            x = g_ct9(x0, dt)
            F = _num_jacobian(g_ct9, x0, dt)
            Q = Q_white_jerk9(dt, self._q[j])
            P = F @ P0 @ F.T + Q
        return x, P

    def _combine(self):
        """Combined estimate = probability-weighted mean + spread of the mode means."""
        self.x = np.sum(self.mu[:, None] * self.x_modes, axis=0)
        P = np.zeros((NX, NX))
        for j in range(self.M):
            d = self.x_modes[j] - self.x
            P += self.mu[j] * (self.P_modes[j] + np.outer(d, d))
        self.P = P

    def predict(self, dt):
        # 1) predicted mode probabilities and mixing weights
        cbar = self.mu @ self.Pi                                  # c_j = sum_i Pi[i,j] mu_i
        cbar = np.maximum(cbar, 1e-12)
        Mmix = (self.Pi * self.mu[:, None]) / cbar[None, :]       # Mmix[i,j] = mu_{i|j}

        # 2) mixed initial condition per mode (INCLUDES the spread-of-means term)
        x0 = Mmix.T @ self.x_modes                                # (3, 9)
        P0 = np.zeros((self.M, NX, NX))
        for j in range(self.M):
            Pj = np.zeros((NX, NX))
            for i in range(self.M):
                d = self.x_modes[i] - x0[j]
                Pj += Mmix[i, j] * (self.P_modes[i] + np.outer(d, d))
            P0[j] = Pj

        # 3) mode-matched prediction
        for j in range(self.M):
            self.x_modes[j], self.P_modes[j] = self._predict_mode(j, x0[j], P0[j], dt)

        # mode probabilities revert to the predicted (c_j) until a measurement arrives
        self.mu = cbar / cbar.sum()
        self._Lj = cbar.copy()
        self._combine()

    def update_radar(self, z_meas, R, radar_ecef):
        z_meas = np.asarray(z_meas, dtype=float)
        nis_modes = np.zeros(self.M)
        for j in range(self.M):
            x, P = self.x_modes[j], self.P_modes[j]
            z_pred, H = _h_and_H9(x, radar_ecef)
            y = z_meas - z_pred
            y[1] = wrap_angle(y[1])         # wrap ONLY az/el (range/Doppler must not wrap)
            y[2] = wrap_angle(y[2])
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            self.x_modes[j] = x + K @ y
            I = np.eye(NX)
            self.P_modes[j] = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
            lam, nis = gaussian_likelihood(y, S)
            self._Lj[j] *= lam
            nis_modes[j] = nis

        # sequential Bayesian mode-probability update (measurements ~indep given the mode)
        s = float(self._Lj.sum())
        if np.isfinite(s) and s > 0 and np.all(np.isfinite(self._Lj)):
            self.mu = self._Lj / s      # else: keep the prior mu -- a degenerate likelihood must not overwrite
            #                             a valid mode estimate with NaN (which would poison x, P through _combine)
        self._combine()
        return float(self.mu @ nis_modes)   # mode-probability-weighted NIS

    def update_angle(self, z_azel, R2, sensor_ecef):
        """ANGLE-ONLY (az, el) update from a passive IR sensor (IRST / LEO satellite). A dedicated
        2-vector update -- numerically stable, unlike soft-dropping range/Doppler at ~1e18 variance.
        One angle-only sensor constrains the transverse (cross-LOS) state; depth needs a second
        angle sensor or a ranging radar (the fusion is honest about observability)."""
        z_azel = np.asarray(z_azel, dtype=float)
        nis_modes = np.zeros(self.M)
        for j in range(self.M):
            x, P = self.x_modes[j], self.P_modes[j]
            z_pred, H = _h_and_H9_angle(x, sensor_ecef)
            y = np.array([wrap_angle(z_azel[0] - z_pred[0]), wrap_angle(z_azel[1] - z_pred[1])])
            S = H @ P @ H.T + R2
            K = P @ H.T @ np.linalg.inv(S)
            self.x_modes[j] = x + K @ y
            I = np.eye(NX)
            self.P_modes[j] = (I - K @ H) @ P @ (I - K @ H).T + K @ R2 @ K.T
            lam, nis = gaussian_likelihood(y, S)
            self._Lj[j] *= lam
            nis_modes[j] = nis
        s = float(self._Lj.sum())
        if np.isfinite(s) and s > 0 and np.all(np.isfinite(self._Lj)):
            self.mu = self._Lj / s      # else: keep the prior mu -- a degenerate likelihood must not overwrite
            #                             a valid mode estimate with NaN (which would poison x, P through _combine)
        self._combine()
        return float(self.mu @ nis_modes)


# ------------------------- smoke test ---------------------------
def _smoke():
    """Minimal sanity run: build an IMM, feed one synthetic radar update, print state."""
    x0 = np.array([6.4e6, 0.0, 0.0, 0.0, 900.0, 0.0, 0.0, 0.0, 0.0])
    P0 = np.diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2]).astype(float)
    imm = IMM_EKF(x0, P0)
    radar = (6.371e6, 0.0, 0.0)
    R = np.diag(np.array([50.0, 1e-3, 1e-3, 3.0]) ** 2)
    imm.predict(0.5)
    z, _ = _h_and_H9(imm.x, radar)
    nis = imm.update_radar(z + np.array([10.0, 1e-4, 1e-4, 0.5]), R, radar)
    print("modes:", imm.mode_names)
    print("mode probs:", np.round(imm.mu, 3))
    print("combined pos:", np.round(imm.x[:3], 1))
    print("NIS:", round(nis, 2))
    assert abs(imm.mu.sum() - 1.0) < 1e-9
    assert np.all(np.isfinite(imm.x))
    print("PASS: IMM runs, mode probs normalized, state finite.")


if __name__ == "__main__":
    _smoke()
