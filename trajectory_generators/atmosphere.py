"""
trajectory_generators/atmosphere.py - US Standard Atmosphere 1976 (layered), replacing the
single-exponential density model. Grounded in the table shipped with MAPLEAF
(impfolder/2/MAPLEAF/ENV/US_STANDARD_ATMOSPHERE.txt); see reference/GROUNDING_MANIFEST.md (A8).

Density is log-interpolated in altitude (smooth, positive). Above the table top (~86 km) an
exponential tail is used (density is negligible there anyway). Also exposes temperature and
sound speed (for Mach).
"""
from __future__ import annotations
import numpy as np

# Altitude (m), density (kg/m^3), temperature (K) — US Standard Atmosphere 1976.
_H = np.array([
    -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000,
    36000, 38000, 40000, 42000, 44000, 46000, 48000, 50000, 52000, 54000, 56000, 58000,
    60000, 62000, 64000, 66000, 68000, 70000, 72000, 74000, 76000, 78000, 80000, 82000,
    84000, 86000], dtype=float)
_RHO = np.array([
    1.478, 1.347, 1.225, 1.112, 1.007, 0.9093, 0.8194, 0.7364, 0.6601, 0.59, 0.5258,
    0.4671, 0.4135, 0.3119, 0.2279, 0.1665, 0.1216, 0.08891, 0.06451, 0.04694, 0.03426,
    0.02508, 0.01841, 0.01355, 0.009887, 0.007257, 0.005366, 0.003995, 0.002995, 0.002259,
    0.001714, 0.001317, 0.001027, 0.0008055, 0.0006389, 0.0005044, 0.0003962, 0.0003096,
    0.0002407, 0.000186, 0.0001429, 0.0001091, 0.00008281, 0.00006236, 0.00004637,
    0.0000343, 0.00002523, 0.00001845, 0.00001341, 0.00000969, 0.000006955], dtype=float)
_T = np.array([
    301.2, 294.65, 288.15, 281.65, 275.15, 268.66, 262.17, 255.68, 249.19, 242.7, 236.21,
    229.73, 223.25, 216.6, 216.6, 216.6, 216.6, 216.6, 218.6, 220.6, 222.5, 224.5, 226.5,
    228.5, 233.7, 239.3, 244.8, 250.4, 255.9, 261.4, 266.9, 270.6, 270.6, 269, 263.5, 258,
    252.5, 247, 241.5, 236, 230.5, 225.1, 219.6, 214.3, 210.3, 206.4, 202.5, 198.6, 194.7,
    190.8, 186.9], dtype=float)

_LOG_RHO = np.log(_RHO)
_TOP_H = _H[-1]
_TOP_RHO = _RHO[-1]
_TAIL_SCALE = 6000.0        # exponential scale height above the table top
GAMMA_AIR = 1.4
R_AIR = 287.05


def density(alt_m: float) -> float:
    """Air density (kg/m^3) at geometric altitude alt_m."""
    if alt_m <= _H[0]:
        return float(_RHO[0])
    if alt_m >= _TOP_H:
        return float(_TOP_RHO * np.exp(-(alt_m - _TOP_H) / _TAIL_SCALE))
    return float(np.exp(np.interp(alt_m, _H, _LOG_RHO)))


def temperature(alt_m: float) -> float:
    """Air temperature (K)."""
    a = min(max(alt_m, _H[0]), _TOP_H)
    return float(np.interp(a, _H, _T))


def sound_speed(alt_m: float) -> float:
    """Speed of sound (m/s) = sqrt(gamma R T)."""
    return float(np.sqrt(GAMMA_AIR * R_AIR * temperature(alt_m)))


if __name__ == "__main__":
    for h in (0, 5000, 14000, 30000, 50000, 80000):
        print("h=%6d m  rho=%.4e kg/m^3  T=%.1f K  a=%.1f m/s"
              % (h, density(h), temperature(h), sound_speed(h)))
