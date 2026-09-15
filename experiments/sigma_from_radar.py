"""experiments/sigma_from_radar.py -- acceleration noise implied by radar position measurements.

A radar measures position, so lateral acceleration comes from differentiating position twice.
For a least-squares quadratic fit to N position samples at spacing T with per-sample error
sigma_r, the acceleration estimate has standard deviation 2 * sigma_r * sqrt([(X'X)^-1]_33).
At N = 3 this is the second difference, sqrt(6) * sigma_r / T^2.

Radar parameters are read from sim/radars.py (50 m range error, 2 Hz update rate).

    python experiments/sigma_from_radar.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                            # noqa: E402

from sim.radars import RADARS                                                 # noqa: E402

TARGET_SIGMA = 0.3      # m/s^2, the comparator noise at the paper's operating point


def sigma_accel(sigma_r, T, N):
    """Standard deviation of the fitted acceleration for a quadratic fit to N samples."""
    t = (np.arange(N) - (N - 1) / 2.0) * T
    X = np.vstack([np.ones(N), t, t ** 2]).T
    return 2.0 * sigma_r * np.sqrt(np.linalg.inv(X.T @ X)[2, 2])


def main():
    radar = RADARS[0]
    sigma_r = float(radar["sigma_range_m"])
    T = 1.0 / float(radar["update_rate_hz"])
    print("radar %s: sigma_range %.1f m, update interval %.2f s" % (radar["id"], sigma_r, T))
    print("second difference (N=3): sigma_a = %.1f m/s^2" % (np.sqrt(6.0) * sigma_r / T ** 2))
    print("\n%6s %10s %14s" % ("N", "span (s)", "sigma_a (m/s^2)"))
    for N in (3, 5, 11, 21, 51):
        print("%6d %10.1f %14.3f" % (N, (N - 1) * T, sigma_accel(sigma_r, T, N)))
    n_min = next(N for N in range(3, 1000) if sigma_accel(sigma_r, T, N) <= TARGET_SIGMA)
    print("\nsmallest N reaching %.1f m/s^2: %d samples spanning %.1f s"
          % (TARGET_SIGMA, n_min, (n_min - 1) * T))


if __name__ == "__main__":
    main()
