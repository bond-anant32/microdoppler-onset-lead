"""
sim/detection.py - probability-of-detection gate for realistic radar (Swerling models).

Real radars miss low-SNR returns. This module turns the SNR our measurement model already
computes into a stochastic detection decision, so the EKF/IMM must cope with missed
detections (predict-only steps) - a core piece of sensor realism (component 3).

Physics is the STANDARD public radar detection theory (Mahafza, "Radar Systems Analysis";
Skolnik): a square-law detector with N non-coherently integrated pulses has threshold
VT = Gamma^{-1}(N, Pfa) (upper-incomplete-gamma inverse), and closed-form Pd per Swerling
fluctuation model. Reimplemented from those public equations (NOT copied from the GPL
RadarSimPy source); the __main__ VALIDATES this independent implementation against
RadarSimPy's pinned reference values (impfolder/1/.../tests/test_module_tools.py).

Swerling 1 (slow, scan-to-scan fluctuation) is the right default for a maneuvering,
aspect-changing missile.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.special import gammainc, gammainccinv   # regularized lower P(a,x); inverse of upper Q


def threshold(pfa: float, n_pulses: int = 1) -> float:
    """Detection threshold VT for N non-coherently integrated pulses at false-alarm prob pfa.
    Solves Q(N, VT) = pfa, i.e. VT = Gamma^{-1}_upper(N, pfa)."""
    return float(gammainccinv(n_pulses, pfa))


def pd_swerling1(pfa: float, snr_db: float, n_pulses: int = 1) -> float:
    """Detection probability, Swerling 1 (Rayleigh, slow fluctuation)."""
    snr = 10.0 ** (snr_db / 10.0)
    VT = threshold(pfa, n_pulses)
    N = n_pulses
    if N == 1:
        return float(np.exp(-VT / (1.0 + snr)))
    return float(1.0 - gammainc(N - 1, VT)
                 + (1.0 + 1.0 / (N * snr)) ** (N - 1)
                 * gammainc(N - 1, VT / (1.0 + 1.0 / (N * snr)))
                 * np.exp(-VT / (1.0 + N * snr)))


def pd_swerling2(pfa: float, snr_db: float, n_pulses: int = 1) -> float:
    """Detection probability, Swerling 2 (Rayleigh, fast pulse-to-pulse fluctuation)."""
    snr = 10.0 ** (snr_db / 10.0)
    VT = threshold(pfa, n_pulses)
    return float(1.0 - gammainc(n_pulses, VT / (1.0 + snr)))


def prob_detection(snr_db: float, pfa: float = 1e-6, n_pulses: int = 1, swerling: int = 1) -> float:
    if swerling == 2:
        return pd_swerling2(pfa, snr_db, n_pulses)
    return pd_swerling1(pfa, snr_db, n_pulses)


def detected(snr_db: float, rng: np.random.Generator,
             pfa: float = 1e-6, n_pulses: int = 1, swerling: int = 1) -> Tuple[bool, float]:
    """Stochastic detection decision. Returns (is_detected, pd)."""
    pd = prob_detection(snr_db, pfa, n_pulses, swerling)
    return bool(rng.random() < pd), pd


if __name__ == "__main__":
    # Validate this independent implementation against RadarSimPy's pinned reference values.
    import numpy.testing as npt
    checks = [
        ("threshold(1e-4, 1)",  threshold(1e-4, 1),  9.21),
        ("threshold(1e-4, 10)", threshold(1e-4, 10), 26.19),
        ("threshold(1e-4, 20)", threshold(1e-4, 20), 41.03),
        ("threshold(1e-4, 40)", threshold(1e-4, 40), 67.89),
        ("Pd Sw1 16dB N1",   pd_swerling1(1e-4, 16, 1),   0.7980),
        ("Pd Sw1 6.8dB N1",  pd_swerling1(1e-4, 6.8, 1),  0.2036),
        ("Pd Sw1 3.6dB N256",  pd_swerling1(1e-4, 3.6, 256),  0.8959),
        ("Pd Sw1 -7.6dB N256", pd_swerling1(1e-4, -7.6, 256), 0.2560),
        ("Pd Sw2 -4.4dB N256", pd_swerling2(1e-4, -4.4, 256), 0.9120),
        ("Pd Sw2 16dB N1",    pd_swerling2(1e-4, 16, 1),      0.7980),
    ]
    print("Validating independent detection theory vs RadarSimPy pinned values:")
    ok = True
    for name, got, expect in checks:
        tol = 0.05 if "threshold" in name else 5e-3
        passed = abs(got - expect) < tol
        ok = ok and passed
        print(f"  {'ok ' if passed else 'XX '} {name:22s} got {got:9.4f}  expect {expect:9.4f}")
    print("\n" + ("ALL PASS - detection theory grounded & validated." if ok
                  else "MISMATCH - inspect formulas."))
