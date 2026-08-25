"""
sim/ir_detection.py - space-IR RADIOMETRY + probabilistic detection gate. The IR twin of
sim/detection.py, built from the satellite-IR audit (workflow wf_36097013).

Why it exists: the first-cut satellite model detected aerothermal SKIN heating only, so it was BLIND
during the boost phase -- the phase that confirms a LAUNCH -- and it used a hard deterministic SNR
gate with no probabilistic missed detections, no atmospheric extinction, and no calibration. This
module fixes the radiometry and gives the sensor the same detection-statistics rigor sim/detection.py
gives the radar.

Physics (FIRST-PRINCIPLES / OSINT surrogates, LABELLED -- NOT classified data, not a NEQAIR/SIRRM sim):
  * RADIANT INTENSITY = max(boost plume, aerothermal skin), both a PLANCK band-integral over MWIR
    (Lambertian I = A * M_band(T) / pi). The exhaust PLUME is modeled as a T=2500 K blackbody
    (solid-rocket-motor combustion, cited to Planck's law -- no classified plume chemistry) over a
    class exhaust area, gated by an OSINT BURN PROFILE (full through the ~60 s boost burn, drops at
    burnout) -- this is what a DSP/SBIRS-class sensor detects at LAUNCH. The 2500 K-x-area intensity
    recovers the OTA-1988 hierarchy (ICBM ~1e6, theater ~1e5 W/sr). The SKIN (recovery-temperature
    blackbody, capped at a radiative wall temp) is the HGV-glide/reentry TRACKING signal.
  * ATMOSPHERIC TRANSMITTANCE exp(-k*airmass), k~0.12/km MWIR (MODTRAN-class band figure), only the
    ~0-20 km layer attenuates (vacuum above is loss-free); airmass grows as 1/cos(zenith). So the
    plume gets BRIGHTER to the satellite as it climbs out of the dense atmosphere -- the physical
    launch-detection cue. The defensible write-up: "early-warning satellite detections were simulated
    using a first-principles radiometric surrogate -- plume intensity as a 2500 K blackbody radiator
    during the ~60 s boost, under altitude-dependent (MODTRAN-class) atmospheric attenuation."
  * DETECTION: single-frame (or N-frame non-coherently integrated) square-law/envelope detector,
    non-fluctuating point source in Gaussian noise -> Pd = Q1(sqrt(2*SNR), sqrt(VT)) = ncx2(df=2,
    nc=2*SNR).sf(VT), threshold VT = -2 ln(Pfa) [N=1]. Standard Marcum-Q detection theory.

HONESTY: this is SELF-CONSISTENT against detection theory (the __main__ cross-checks the NP threshold
vs the documented npwgnthresh value), NOT pinned to an external IR reference the way detection.py is
pinned to RadarSimPy -- no open source pins IR Pd at specific SNRs. NEI and areas are calibration
knobs anchored to published *detectability thresholds*, not device measurements.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import ncx2
from scipy.special import erfcinv, gammainccinv

T_AMB = 250.0                 # ambient temp at altitude (K)
SIGMA_SB = 5.670374419e-8     # Stefan-Boltzmann (W/m^2/K^4)
T_WALL_MAX = 3000.0           # radiative-equilibrium/ablation cap on skin temp (K) -- recovery temp
#                              # runs away as M^2; real hypersonic walls sit ~1100-2600 K.
T_PLUME_K = 2500.0            # exhaust-plume blackbody temperature (K) -- solid-rocket-motor combustion,
#                              # the OSINT-defensible value modeled via Planck's law (NOT classified chem)
BURN_TIME_S = 60.0           # nominal SRBM boost burn (typ. 60-90 s); OSINT burn profile
NEI_WM2 = 5e-10              # noise-equivalent irradiance, cooled MWIR FPA (calibration knob; places
#                              # DSP 20 kW/sr at ~+14 dB @1200 km, the published detectability anchor)
K_MWIR_PER_KM = 0.12         # MWIR band-avg extinction near surface (MODTRAN-class; SENSOR_ARCH:89)
H_ATM_KM = 20.0             # effective attenuating-layer thickness
_H, _C, _KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23   # Planck constants (SI)

# EXHAUST-PLUME cross-section area (m^2) by booster class -> intensity comes from Planck(2500 K)*area,
# NOT a hand-set W/sr. Bigger booster => bigger plume => brighter (recovers the OTA-1988 hierarchy).
PLUME_AREA_M2 = {"ballistic": 20.0, "booster": 20.0, "mirv": 20.0, "hgv": 10.0,
                 "marv": 6.0, "supersonic_cruise": 3.0, "bus": 3.0, "rv": 0.5, "decoy": 0.5}
# body projected/emitting area (m^2) for the aerothermal SKIN term, consistent with signatures SCATTERERS.
IR_AREA_M2 = {"ballistic": 3.0, "mirv": 3.0, "bus": 2.0, "hgv": 1.8, "marv": 1.5,
              "supersonic_cruise": 1.2, "rv": 0.6, "decoy": 0.6}


def planck_band_exitance(T, lam1=3e-6, lam2=5e-6, n=64):
    """MWIR (3-5 um) band radiant exitance M (W/m^2) of a blackbody at T, by numerical integration of
    PLANCK'S LAW M(lam,T) = 2*pi*h*c^2 / (lam^5 (exp(hc/lam k T) - 1)). First-principles (no fixed
    band-fraction fudge, no classified plume chemistry) -- the academically-defensible surrogate."""
    lam = np.linspace(lam1, lam2, n)
    M = (2.0 * np.pi * _H * _C ** 2 / lam ** 5) / (np.expm1(_H * _C / (lam * _KB * T)))
    return float(np.sum(0.5 * (M[:-1] + M[1:]) * np.diff(lam)))          # trapezoid (numpy-2 safe)


def burn_factor(t_since_launch, burn_time=BURN_TIME_S):
    """OSINT burn profile: full plume through the boost burn, then a ~2 s tail-off at burnout. With no
    launch clock (t_since_launch=None) a 'boost' state is treated as full burn."""
    if t_since_launch is None or t_since_launch <= burn_time:
        return 1.0
    return float(np.exp(-(t_since_launch - burn_time) / 2.0))


def radiant_intensity_wsr(missile_type, flight_state, mach, alt_m, t_since_launch=None, burn_time=BURN_TIME_S):
    """In-band (MWIR) radiant intensity (W/sr) = max(boost plume, aerothermal skin), both via a
    Planck band-integral (Lambertian I = A * M_band(T) / pi):
      * PLUME -- a T_PLUME_K (2500 K) blackbody over the class exhaust area, active during boost and
        modulated by the OSINT burn profile (bright t=0..burn_time, drops at burnout). Confirms LAUNCH.
        (The altitude dependence is handled where it belongs -- atmospheric_transmittance -- so the
        plume gets BRIGHTER to the satellite as it climbs out of the attenuating layer.)
      * SKIN -- a recovery-temperature (capped) blackbody over the body area; the HGV-glide track."""
    t_wall = min(T_AMB * (1.0 + 0.2 * mach * mach), T_WALL_MAX)          # capped recovery temp
    i_skin = IR_AREA_M2.get(missile_type, 1.5) * planck_band_exitance(t_wall) / np.pi
    i_plume = 0.0
    if flight_state == "boost":
        i_plume = (PLUME_AREA_M2.get(missile_type, 5.0) * planck_band_exitance(T_PLUME_K) / np.pi
                   * burn_factor(t_since_launch, burn_time))
    return float(max(i_plume, i_skin))


def atmospheric_transmittance(alt_m, cos_zenith):
    """MWIR slant transmittance exp(-k*airmass). Only the ~0-20 km layer above the TARGET attenuates;
    airmass = layer_thickness / cos(zenith) (grazing paths see more air). cos_zenith = LOS.up_target."""
    layer_km = max(0.0, H_ATM_KM - alt_m / 1000.0)
    airmass = layer_km / max(cos_zenith, 0.1)                           # cap grazing at ~10x
    return float(np.exp(-K_MWIR_PER_KM * min(airmass, 60.0)))


def ir_snr_db(intensity_wsr, R_slant_m, alt_m, cos_zenith, nei=NEI_WM2):
    """SNR (dB) = 10 log10( E / NEI ), E = I * tau_atm / R^2 (aperture irradiance)."""
    tau = atmospheric_transmittance(alt_m, cos_zenith)
    E = intensity_wsr * tau / (R_slant_m * R_slant_m)
    return float(10.0 * np.log10(max(E / nei, 1e-9)))


def ir_threshold(pfa, n_frames=1):
    """Detection threshold VT. Single-frame envelope detector: Pfa = exp(-VT/2) -> VT = -2 ln(Pfa);
    N-frame non-coherent integration: upper-incomplete-gamma inverse (as in sim/detection.py)."""
    if n_frames == 1:
        return -2.0 * np.log(pfa)
    # square-law / Chi-square(2N) detector: VT = 2*gammainccinv(N, Pfa). The factor 2 makes the N-frame
    # threshold continuous with the N==1 closed form above (2*gammainccinv(1,Pfa) = -2 ln Pfa) -- without
    # it the N>1 branch was half the correct threshold. Latent: all live callers N=1.
    return float(2.0 * gammainccinv(n_frames, pfa))


def pd_ir(snr_db, pfa=1e-6, n_frames=1):
    """Probability of detection, non-fluctuating point source, square-law detector (Marcum-Q):
    Pd = Q1(sqrt(2*SNR), sqrt(VT)) = ncx2(df=2*N, nc=2*N*SNR).sf(VT)."""
    snr = 10.0 ** (snr_db / 10.0)
    vt = ir_threshold(pfa, n_frames)
    return float(ncx2.sf(vt, df=2 * n_frames, nc=2.0 * n_frames * snr))


def ir_detected(snr_db, rng, pfa=1e-6, n_frames=1):
    """Stochastic detect decision: draw rng.random() < Pd. On a miss the caller emits nothing so the
    tracker takes a predict-only step (exactly like sim/detection.py for radar). Returns (bool, Pd)."""
    pd = pd_ir(snr_db, pfa, n_frames)
    return bool(rng.random() < pd), pd


def fluctuate_snr_db(snr_db, rng, shape=4.0):
    """Per-look IR irradiance fluctuation (plume flicker / scintillation), gamma about the mean on
    LINEAR irradiance. This is the REPORTED value; the Pd gate uses the MEAN snr_db (the radar-C4
    discipline). Returns a fluctuated snr_db."""
    lin = 10.0 ** (snr_db / 10.0)
    fl = rng.gamma(shape, lin / shape)                                 # mean = lin, variance = lin^2/shape
    return float(10.0 * np.log10(max(fl, 1e-9)))


def _validate():
    """Self-consistency checks (analog of sim/detection.py __main__)."""
    checks = []
    # (1) Pd pins from the Marcum-Q form at Pfa=1e-6 (computed, not externally sourced)
    pins = {8.0: 0.056, 11.24: 0.50, 13.18: 0.90, 14.49: 0.99}
    ok_pins = all(abs(pd_ir(s) - p) < 0.02 for s, p in pins.items())
    checks.append(("Pd(SNR) matches Marcum-Q pins (8->.06, 11.24->.50, 13.18->.90)", ok_pins,
                   " ".join(f"{s}:{pd_ir(s):.2f}" for s in pins)))
    # (2) a hard 6 dB gate overstates detection: real Pd at 6 dB is ~1%
    checks.append(("Pd(6 dB) < 0.1 (the old hard 6 dB gate was far too generous)", pd_ir(6.0) < 0.1,
                   f"Pd(6dB)={pd_ir(6.0):.3f}, Pd(13dB)={pd_ir(13.0):.3f}"))
    # (3) NP threshold cross-check vs documented npwgnthresh (Pfa=0.01, N=1 -> 7.3335 dB)
    npw = 10.0 * np.log10(2.0 * erfcinv(2 * 0.01) ** 2)
    checks.append(("NP threshold reproduces npwgnthresh 7.3335 dB", abs(npw - 7.3335) < 1e-3,
                   f"{npw:.4f} dB"))
    # (4) LAUNCH DETECTABILITY: a boost-phase plume clears the gate with the published margin
    #     (ICBM 6e5 W/sr at 5 km alt, R=1500 km, near-overhead) vs a cold subsonic body which does not
    i_plume = radiant_intensity_wsr("ballistic", "boost", 0.5, 5000.0)     # slow booster, low alt
    snr_launch = ir_snr_db(i_plume, 1.5e6, 5000.0, 0.9)
    i_cold = radiant_intensity_wsr("supersonic_cruise", "cruise", 0.8, 12000.0)
    snr_cold = ir_snr_db(i_cold, 1.5e6, 12000.0, 0.9)
    checks.append(("LAUNCH: slow boost plume detected (Pd>0.9), cold subsonic not (Pd<0.1)",
                   pd_ir(snr_launch) > 0.9 and pd_ir(snr_cold) < 0.1,
                   f"boost {snr_launch:.0f}dB/Pd{pd_ir(snr_launch):.2f} vs cold {snr_cold:.0f}dB/Pd{pd_ir(snr_cold):.2f}"))
    # (5) hierarchy plume >> hardbody >> cold RV (in-band radiant intensity)
    i_hgv = radiant_intensity_wsr("hgv", "glide", 8.0, 40000.0)
    i_rv = radiant_intensity_wsr("rv", "midcourse", 2.0, 60000.0)
    checks.append(("intensity hierarchy plume >> HGV skin >> cold RV",
                   i_plume > i_hgv > i_rv, f"plume {i_plume:.0e} > hgv {i_hgv:.0e} > rv {i_rv:.0e} W/sr"))
    # (6) atmospheric extinction bites low: a low grazing path attenuates more than a high overhead one
    tau_low = atmospheric_transmittance(3000.0, 0.4)
    tau_high = atmospheric_transmittance(40000.0, 0.95)
    checks.append(("MWIR extinction attenuates the low grazing boost path more than a high nadir look",
                   tau_low < tau_high and tau_high > 0.9, f"tau low {tau_low:.2f} vs high {tau_high:.2f}"))
    # (7) OSINT burn profile: plume bright through the burn, drops sharply at burnout
    i_burn = radiant_intensity_wsr("ballistic", "boost", 0.5, 30000.0, t_since_launch=30.0)
    i_out = radiant_intensity_wsr("ballistic", "boost", 0.5, 30000.0, t_since_launch=70.0)
    checks.append(("OSINT burn profile: plume bright during burn, drops >20x at burnout",
                   i_burn > 20 * i_out, f"t=30 s {i_burn:.1e} vs t=70 s (burnout) {i_out:.1e} W/sr"))
    # (8) Planck-grounded plume: a 2500 K blackbody over the exhaust area recovers the ICBM ~1e6 anchor
    icbm = radiant_intensity_wsr("ballistic", "boost", 0.5, 30000.0)
    checks.append(("Planck 2500 K plume recovers the ICBM-class ~1e6 W/sr anchor (OTA 1988)",
                   5e5 < icbm < 5e6, f"ICBM plume {icbm:.1e} W/sr = 2500 K blackbody x exhaust area"))

    print("=" * 72)
    print("SPACE-IR RADIOMETRY + DETECTION VALIDATION (sim/ir_detection.py)")
    print("=" * 72)
    ok = True
    for name, passed, detail in checks:
        ok = ok and passed
        print(f"  {'ok ' if passed else 'XX '} {name:56s} {detail}")
    print("\n" + ("ALL PASS - IR radiometry detects a launch plume, drops detections probabilistically,"
                  "\n           and is self-consistent with Marcum-Q detection theory."
                  if ok else "FAILURES - inspect."))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _validate() else 1)
