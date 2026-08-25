"""
sim/signatures.py - Grounded multi-modal radar signature synthesizer.

HRR, micro-Doppler, and RCS are all generated
from ONE physical model: a per-class set of scattering centers with micro-motions, driven by
CONTINUOUS physical state (aspect, Mach, lateral acceleration, and a control-surface
deflection cue that LEADS the kinematic turn). Nothing is gated on the label.

All four impfolder tools are aggregated here:
  * OptixRCS (impfolder/3): real aspect-resolved RCS maps (X-35) calibrate the flash/null
    dynamic range; the trihedral reflector is the analytic RCS unit test.
  * RadarSimPy (impfolder/1): its real range_fft / doppler_fft DSP synthesize the HRR (range
    compression) and micro-Doppler (STFT) from the scattering-center returns. Loaded via a
    shim that bypasses the dead C++ engine; a pure-NumPy windowed-FFT fallback is used if the
    import ever fails, so the pipeline never breaks.
  * Fusion-DeepONet (impfolder/4): its bow-shock pressure field grounds the HGV plasma-sheath
    effect (deterministic RCS bias + Doppler broadening at high Mach / low altitude).
  * MAPLEAF (impfolder/2): the layered atmosphere (already in trajectory_generators/atmosphere)
    and NASA 6-DOF attitude (the EKF/IMM validation set) come from here.

The anticipation mechanism (the thesis): a maneuver is preceded by control-surface deflection.
The actual command->kinematic lag is trajectory_generators.maneuvering.ACTUATOR_LAG_S = 1.2 s
(the generator applies the recorded command with that delay); the micro-Doppler of the
control-surface scatterer tracks the command, so the signature carries a LEADING cue that a
kinematic filter cannot see.
"""
from __future__ import annotations
import os
import sys
import math
import types
import importlib.util
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

C_LIGHT = 299_792_458.0
FC_HZ = 10.0e9            # X-band nominal carrier
BANDWIDTH_HZ = 1.0e9      # 1 GHz -> range resolution c/2B = 0.15 m
LAMBDA = C_LIGHT / FC_HZ
LEAD_S = 1.5              # LEGACY doc constant, functionally UNUSED: the real
                          # lag is maneuvering.ACTUATOR_LAG_S = 1.2 s. Kept only so old experiment
                          # docstrings (train_onset/onset_uplift) don't dangle; do not cite 1.5 s.

# ---------------------------------------------------------------------------
# Tool 1: RadarSimPy real DSP (range_fft / doppler_fft), loaded past the dead engine.
# ---------------------------------------------------------------------------
def _load_radarsimpy_dsp():
    base = os.path.join(_ROOT, "impfolder", "1", "radarsimpy-master", "src", "radarsimpy")
    if not os.path.isdir(base):
        return None
    try:
        if "radarsimpy" not in sys.modules:
            pkg = types.ModuleType("radarsimpy")
            pkg.__path__ = [base]
            sys.modules["radarsimpy"] = pkg
        for name in ("tools", "processing"):
            key = f"radarsimpy.{name}"
            if key not in sys.modules:
                spec = importlib.util.spec_from_file_location(key, os.path.join(base, f"{name}.py"))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[key] = mod
                spec.loader.exec_module(mod)
        return sys.modules["radarsimpy.processing"]
    except Exception:
        return None

_RSP = _load_radarsimpy_dsp()
RADARSIMPY_DSP = _RSP is not None


def _range_fft(fast_time: np.ndarray) -> np.ndarray:
    """Range compression of a fast-time return -> HRR (magnitude). Uses RadarSimPy's real
    range_fft (cube axis=2) when available, else an equivalent Hann-windowed FFT."""
    x = np.asarray(fast_time, dtype=complex)
    if _RSP is not None:
        cube = x.reshape(1, 1, -1)
        win = np.hanning(cube.shape[2])
        out = _RSP.range_fft(cube, rwin=win)
        return np.abs(out).reshape(-1)
    win = np.hanning(x.size)
    return np.abs(np.fft.fft(x * win))


def _doppler_fft(slow_time_window: np.ndarray) -> np.ndarray:
    """One Doppler spectrum of a slow-time window. Uses RadarSimPy's real doppler_fft
    (cube axis=1) when available, else a Hann-windowed FFT + fftshift."""
    x = np.asarray(slow_time_window, dtype=complex)
    if _RSP is not None:
        cube = x.reshape(1, -1, 1)
        win = np.hanning(cube.shape[1])
        out = _RSP.doppler_fft(cube, dwin=win)
        return np.fft.fftshift(np.abs(out).reshape(-1))   # center zero-Doppler (RSP returns DC at index 0, unshifted)
    win = np.hanning(x.size)
    return np.abs(np.fft.fftshift(np.fft.fft(x * win)))


# ---------------------------------------------------------------------------
# Tool 2: OptixRCS X-35 aspect map -> calibrate the flash/null dynamic range.
# ---------------------------------------------------------------------------
def _load_x35_dynamic_range() -> float:
    """Return the broadside-flash minus median (dB) of the real X-35 aspect map, used to scale
    our synthetic aspect dependence to a physically-observed dynamic range. Falls back to 30 dB
    (typical airframe flash/null span) if the file is missing."""
    path = os.path.join(_ROOT, "impfolder", "3", "output", "x35_rcs.csv")
    try:
        vals = np.genfromtxt(path, delimiter=",", usecols=3)
        vals = vals[np.isfinite(vals)]
        if vals.size < 100:
            return 30.0
        return float(np.percentile(vals, 99) - np.percentile(vals, 50))
    except Exception:
        return 30.0

X35_DYNAMIC_RANGE_DB = _load_x35_dynamic_range()


# ---------------------------------------------------------------------------
# Tool 3: Fusion-DeepONet bow-shock pressure -> HGV plasma-sheath factor.
# ---------------------------------------------------------------------------
def _load_plasma_pressure_ratio() -> float:
    """Peak/inflow static-pressure ratio across the bow shock from the DeepONet flow field.
    Grounds the magnitude of the plasma-sheath effect (attenuation + Doppler broadening) applied
    to HGV signatures at high Mach / low altitude. Falls back to ~130 (from the manifest) if
    the npz is unavailable."""
    path = os.path.join(_ROOT, "impfolder", "4", "Fusion-DeepONet-main", "src",
                        "Semi_ellipse", "structured_grid", "Fusion_DeepONet_preds.npz")
    try:
        d = np.load(path)
        arr = d["target"]                       # (8, 256, 256, 4) -> last channel ~ pressure
        p = arr[..., -1]
        pr = float(np.nanmax(p) / max(np.nanpercentile(p, 5), 1e-6))
        return float(np.clip(pr, 10.0, 500.0))
    except Exception:
        return 130.0

PLASMA_PRESSURE_RATIO = _load_plasma_pressure_ratio()


# ---------------------------------------------------------------------------
# Scattering-center geometry per class.
#   down_range_m : offset along the body axis from the nose (m)
#   base_dbsm    : nominal RCS of this center (dBsm)
#   kind         : 'tip' (aspect-broad, small), 'cavity' (corner-reflector: broad, nose-biased),
#                  'cylinder' (broadside specular flash), 'fin'/'control' (small, micro-motion)
# Grounded in rcs-anchors.md (DF-15 head-on ~-17 dBsm; ramjet intake cavity lifts nose-on;
# cylinder broadside flash >> nose-on) and calibrated against the OptixRCS X-35 dynamic range.
# ---------------------------------------------------------------------------
# --- Canonical-analog physical-optics RCS --------------------
# Defense practice (DTIC/AFIT/IEEE Aerospace CEM): model a classified target as a set of
# CANONICAL GEOMETRIC ANALOGS with ANALYTIC physical-optics RCS, because exact RCS is classified.
# Each scattering center is one canonical shape with REAL dimensions; its RCS is computed from the
# textbook PO formula at the operating wavelength, with true specular lobing (sinc^2), NOT a
# hand-set constant. po_rcs_dbsm() is validated against the exact impfolder anchors
# (sphere -1.05 dBsm, plate 48.3/59.2 dBsm, trihedral ~32 dBsm) in tests/test_signatures.py.
#   theta convention HERE (po_rcs_lin's own): 0 = the analog's BORESIGHT along the body long-axis, pi/2 = broadside.
#   NOTE the old note read "0 = nose-on/tail-on", conflating the two ends of the axis. That is harmless for every
#   SYMMETRIC lobe (plate/cylinder sinc^2(k*L*cos t), cone_tip |sin t| -- all invariant under t->pi-t) but NOT for the
#   asymmetric trihedral, and the conflation is what hid the inverted intake-cavity return. The CALLER's aspect
#   (measurements._aspect_angle_rad) uses the opposite sense -- pi = nose-on -- and _center_rcs_lin bridges the two.
def po_rcs_lin(shape: str, dims: Dict[str, float], theta: float, lam: float) -> float:
    """Monostatic physical-optics RCS (linear m^2) of a canonical analog at aspect theta.
    Pure specular PO (no diffuse floor) so it reproduces the EXACT anchors; the small structural
    diffuse return is added separately in _center_rcs_lin for the missile-body sum."""
    k = 2.0 * math.pi / lam
    if shape == "sphere":                                   # optical: sigma = pi a^2
        a = dims["r"]
        return math.pi * a * a
    if shape == "plate":                                    # sigma = 4 pi A^2/lam^2, sinc^2 off-normal
        A = dims["A"]; L = math.sqrt(A)
        smax = 4.0 * math.pi * A * A / (lam * lam)
        arg = k * L * math.cos(theta)                       # normal along body axis -> flash near nose
        s = 1.0 if abs(arg) < 1e-6 else math.sin(arg) / arg
        return smax * s * s
    if shape == "cylinder":                                 # broadside flash 2 pi r L^2/lam, sinc^2
        r = dims["r"]; L = dims["L"]
        smax = 2.0 * math.pi * r * L * L / lam
        arg = k * L * math.cos(theta)                       # flash at broadside (theta=pi/2 -> cos=0)
        s = 1.0 if abs(arg) < 1e-6 else math.sin(arg) / arg
        return smax * s * s
    if shape == "trihedral":                                # corner reflector: 4 pi a^4/3 lam^2, broad
        a = dims["a"]; bw = dims.get("bw", 0.7)
        smax = 4.0 * math.pi * a ** 4 / (3.0 * lam * lam)
        return smax * math.exp(-(theta / bw) ** 2)          # broad retroreflective lobe, nose-biased
    if shape == "cone_tip":                                 # nose-on tip diffraction: low, weakly dep.
        return dims.get("sigma", 1e-2) * (0.3 + abs(math.sin(theta)))
    return 1e-3

def _diffuse_lin(shape: str, dims: Dict[str, float]) -> float:
    """Small ABSOLUTE structural/diffuse RCS (m^2), independent of the razor-thin specular max, so
    off-flash aspects sit at a realistic low level (~-20..-8 dBsm) instead of the specular floor."""
    if shape == "cylinder":
        return 0.02 * 2.0 * dims["r"] * dims["L"]
    if shape == "plate":
        return 0.02 * dims["A"]
    if shape == "trihedral":
        return 0.02 * dims["a"] ** 2
    return 0.0                                              # sphere/cone_tip: PO term already complete

def po_rcs_dbsm(shape: str, dims: Dict[str, float], theta: float = 0.0,
                freq_hz: float = FC_HZ) -> float:
    """Public canonical-analog PO RCS in dBsm (used by the non-circular unit test)."""
    return 10.0 * math.log10(max(po_rcs_lin(shape, dims, theta, C_LIGHT / freq_hz), 1e-12))


@dataclass
class Center:
    down_range_m: float          # offset along the body axis from the nose (m)
    shape: str                   # canonical analog: sphere/plate/cylinder/trihedral/cone_tip
    dims: Dict[str, float]       # real dimensions for the PO formula (r,L,a,A,sigma)
    micro: str                   # micro-motion role: tip/cavity/cylinder/control/fin

# Per-class scattering centers as CANONICAL ANALOGS with real dimensions (metres). RCS follows
# from po_rcs_lin at X-band; nose-on stays low (cone tip + off-broadside nulls), broadside gives
# the specular flash -> the wide flash/null dynamic range real targets show.
SCATTERERS: Dict[str, List[Center]] = {
    "supersonic_cruise": [   # BrahMos/Oniks ~8 m body, 0.6 m dia, ramjet intake cavity
        Center(0.0, "cone_tip", {"sigma": 1.5e-2}, "tip"),
        Center(2.0, "trihedral", {"a": 0.33, "bw": 0.8}, "cavity"),   # engine-inlet corner reflector
        Center(4.5, "cylinder", {"r": 0.30, "L": 4.0}, "cylinder"),
        Center(7.0, "plate", {"A": 0.25}, "control")],
    "ballistic": [           # DF-15 body ~10 m, 1 m dia
        Center(0.0, "cone_tip", {"sigma": 1.0e-2}, "tip"),
        Center(5.0, "cylinder", {"r": 0.50, "L": 6.0}, "cylinder"),
        Center(10.0, "plate", {"A": 0.60}, "fin")],
    "marv": [                # maneuvering RV ~2 m cone, 0.5 m base, control flaps
        Center(0.0, "cone_tip", {"sigma": 3.0e-3}, "tip"),           # stealthy nose
        Center(1.2, "control" if False else "plate", {"A": 0.05}, "control"),
        Center(1.8, "cylinder", {"r": 0.22, "L": 1.4}, "cylinder")],
    "hgv": [                 # waverider ~5 m, sharp leading edge, flat underside, + plasma
        Center(0.0, "cylinder", {"r": 0.05, "L": 3.0}, "tip"),       # thin sharp leading edge
        Center(2.5, "plate", {"A": 1.8}, "cylinder"),                # flat underside
        Center(5.0, "plate", {"A": 0.08}, "control")],
    # --- MIRV objects (bus dispenses these) ---
    "bus": [                 # post-boost vehicle ~3 m, larger body
        Center(0.0, "cone_tip", {"sigma": 1.2e-2}, "tip"),
        Center(1.5, "cylinder", {"r": 0.40, "L": 2.5}, "cylinder"),
        Center(3.0, "plate", {"A": 0.30}, "fin")],
    "rv": [                  # reentry vehicle ~1.5 m sharp cone, spin-stabilized; low nose-on but
        Center(0.0, "cone_tip", {"sigma": 1.2e-2}, "tip"),           # tumbling reentry flashes broadside
        Center(1.0, "cylinder", {"r": 0.28, "L": 1.4}, "cylinder")],
    "decoy": [               # light decoy: mimics the RV's RCS on purpose (drag, not RCS, discriminates)
        Center(0.0, "cone_tip", {"sigma": 1.2e-2}, "tip"),
        Center(0.9, "cylinder", {"r": 0.28, "L": 1.2}, "fin")],
    # --- decoy VARIETY (slice #6): a threat deploys several decoy TYPES with different physics, spanning a
    #     discrimination-difficulty gradient. beta (ballistic coeff) set in sim/mirv.py; SPECS = radar shape. ---
    "decoy_replica": [       # HARD: a faithful RV replica -- matched RCS shape AND near-RV beta; only a
        Center(0.0, "cone_tip", {"sigma": 1.2e-2}, "tip"),           # subtle micro-motion difference betrays it
        Center(1.0, "cylinder", {"r": 0.28, "L": 1.4}, "cylinder")],
    "decoy_balloon": [       # MEDIUM: inflated light balloon -- large broadside RCS, very low beta (falls behind)
        Center(0.0, "plate", {"A": 1.2}, "cylinder"),
        Center(0.6, "plate", {"A": 0.8}, "fin")],
    "decoy_chaff": [         # EASY: dipole cloud -- small fluctuating RCS, extreme drag, broadband micro-Doppler
        Center(0.0, "plate", {"A": 0.06}, "fin"),
        Center(0.4, "plate", {"A": 0.05}, "control"),
        Center(0.8, "plate", {"A": 0.04}, "cylinder")],
}
# Default micro-motion frequencies (Hz) of the moving scatterers, per class.
MICRO_HZ: Dict[str, Dict[str, float]] = {
    "supersonic_cruise": {"engine": 90.0, "control": 6.0},   # inlet spool tone + control flutter
    "ballistic":         {"tumble": 0.3},                    # slow tumble
    "marv":              {"spin": 3.5, "precession": 0.5},   # spin + coning
    "hgv":               {"roll": 1.2},                      # bank-to-turn roll (NOT spin)
    "bus":               {"tumble": 0.4},                    # slow post-boost tumble
    "rv":                {"spin": 2.0, "precession": 0.3},   # spin-stabilized RV
    "decoy":             {"tumble": 1.5},                    # light decoy tumbles faster (a µD cue)
    "decoy_replica":     {"spin": 2.0, "precession": 0.55},  # HARD: nearly mimics RV spin (slightly off precession)
    "decoy_balloon":     {"tumble": 0.6},                    # MEDIUM: big light object, slow tumble
    "decoy_chaff":       {"tumble": 10.0},                   # EASY: dipole cloud, fast/broadband micro-Doppler
}


@dataclass
class Control:
    """Continuous physical drivers for the signature at one instant."""
    lat_accel_mps2: float = 0.0      # current KINEMATIC lateral accel (from velocity) = the label
    lead_cue: float = 0.0            # 0..1 commanded control deflection; leads the turn via actuator lag
    plasma: float = 0.0              # 0..1 plasma-sheath strength (HGV high-Mach/low-alt)
    aoa_rad: float = 0.0             # angle-of-attack proxy: body axis tilts off velocity


def _effective_aspect(aspect_rad: float, aoa_rad: float) -> float:
    """Aspect between the LOS and the BODY axis, which is tilted off the velocity by the
    angle-of-attack. During a hard turn this shifts the RCS/HRR aspect gain.
    Use |aspect - aoa|: when the AoA rotates the body axis PAST the LOS (aoa > aspect near nose-on),
    the LOS is on the other side of the body axis at magnitude aoa-aspect. Clamping to 0.0 instead
    pinned it at nose-on and erased the AoA influence there."""
    return min(abs(aspect_rad - aoa_rad), math.pi)


def _center_rcs_lin(ctr: "Center", aspect_rad: float) -> float:
    """Linear-m^2 RCS of one scattering center: canonical-analog specular PO + small structural
    diffuse floor (so nulls sit at a realistic level, giving the wide flash/null dynamic range).

    ASPECT CONVENTION BRIDGE (bug fix): the CALLER's aspect (sim/measurements._aspect_angle_rad, which builds the
    LOS as radar->target) is pi when the missile flies AT the radar (nose-on) and 0 when it recedes (tail-on).
    po_rcs_lin's own convention is 0 = the analog's BORESIGHT (that is what tests/test_signatures.py anchors the
    trihedral against at theta=0 ~ 32 dBsm), so a shape whose lobe is ASYMMETRIC over [0, pi] must be mapped in.
    Only the TRIHEDRAL is: plate/cylinder use sinc^2(k*L*cos t) and cone_tip |sin t|, all symmetric under t->pi-t,
    so the long-standing "0 = nose-on/tail-on" note above is harmless for them -- and that is exactly why this
    survived: the trihedral is the single shape the conflation can bite, and the regression test uses "ballistic",
    which has no trihedral.
    Left unmapped, the supersonic ramjet ENGINE-INLET cavity (a forward-facing retroreflector, down_range 2.0 m on
    an ~8 m body) flashed TAIL-ON and nulled NOSE-ON -- inverting the intake return by 67 dB and the class total by
    26.5 dB, i.e. the radar saw a DEPARTING cruise missile 4.6x farther than an inbound one (measured end-to-end:
    inbound Pd 0.04 vs outbound Pd 0.99). Four separate comments state the opposite intent ("intake cavity lifts
    nose-on"); this makes the code match them."""
    a = (math.pi - aspect_rad) if ctr.shape == "trihedral" else aspect_rad
    return po_rcs_lin(ctr.shape, ctr.dims, a, LAMBDA) + _diffuse_lin(ctr.shape, ctr.dims)


def rcs_dbsm(missile_type: str, aspect_rad: float, alt_m_true: float, mach: float,
             flight_state: str, control: Optional[Control] = None,
             rng: Optional[np.random.Generator] = None) -> float:
    """Aspect-summed RCS (dBsm) from the canonical-analog scattering centers (analytic PO, true
    specular lobing). Uses the TRUE geodetic altitude. Adds sea-skim multipath,
    control-deflection lift, and (HGV) plasma-sheath bias."""
    centers = SCATTERERS.get(missile_type, SCATTERERS["supersonic_cruise"])
    control = control or Control()
    eff = _effective_aspect(aspect_rad, control.aoa_rad)   # body axis tilted by AoA (H7)
    # Coherent-power sum of the canonical-analog PO returns (linear m^2).
    lin = sum(_center_rcs_lin(ctr, eff) for ctr in centers)
    base = 10.0 * math.log10(max(lin, 1e-9))
    # Altitude (true): sea-skim multipath lift; slight high-altitude reduction.
    if alt_m_true < 100.0:
        base += 2.0
    elif alt_m_true > 15000.0:
        base -= 1.0
    # Exposed control surfaces during a maneuver raise RCS continuously (not a label step).
    base += 2.0 * control.lead_cue
    # HGV plasma sheath: deterministic bias grounded in the DeepONet bow-shock ratio.
    if control.plasma > 0.0:
        base += control.plasma * 2.0 * math.log10(PLASMA_PRESSURE_RATIO)  # ~ +4.5 dB at full plasma
    if rng is not None:
        base += rng.normal(0.0, 0.5)      # residual scintillation (Swerling handled in measurements)
    return float(base)


def hrr_profile(missile_type: str, aspect_rad: float, mach: float,
                control: Optional[Control] = None, n_bins: int = 128,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """High-Range-Resolution profile via range compression of the scattering-center return
    (RadarSimPy range_fft). Peaks sit at each center's DOWN-RANGE projection = offset*cos(aspect),
    so peak spacing is a CONTINUOUS function of aspect (not a class template)."""
    centers = SCATTERERS.get(missile_type, SCATTERERS["supersonic_cruise"])
    control = control or Control()
    rng = rng or np.random.default_rng()
    eff = _effective_aspect(aspect_rad, control.aoa_rad)   # AoA-tilted body aspect (H7)
    n_freq = n_bins
    freqs = np.linspace(FC_HZ - BANDWIDTH_HZ / 2, FC_HZ + BANDWIDTH_HZ / 2, n_freq)
    resp = np.zeros(n_freq, dtype=complex)
    for ctr in centers:
        amp = math.sqrt(max(_center_rcs_lin(ctr, eff), 1e-12))   # PO amplitude (sqrt of RCS)
        if ctr.micro in ("control", "fin"):
            amp *= (1.0 + 1.5 * control.lead_cue)     # control surfaces brighten as they deflect
        r = ctr.down_range_m * math.cos(eff)          # down-range projection along body axis
        resp += amp * np.exp(-1j * 4.0 * math.pi * freqs * r / C_LIGHT)
    resp += (0.02 * (rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)))
    hrr = _range_fft(resp)
    hrr = np.fft.fftshift(hrr)                        # center the zero-range bin
    m = hrr.max()
    return (hrr / m).astype(np.float32) if m > 0 else hrr.astype(np.float32)


def micro_doppler(missile_type: str, aspect_rad: float, mach: float, flight_state: str,
                  control: Optional[Control] = None, H: int = 64, W: int = 64,
                  rng: Optional[np.random.Generator] = None, noise_scale: float = 1.0) -> np.ndarray:
    """Micro-Doppler spectrogram (H freq x W time) via STFT of the summed scattering-center
    micro-motion returns (RadarSimPy doppler_fft per window). The control-surface scatterer's
    modulation tracks `control.lead_cue`, so a MANEUVER APPEARS IN THE SPECTROGRAM BEFORE it is
    kinematically visible. Every class has non-trivial structure."""
    centers = SCATTERERS.get(missile_type, SCATTERERS["supersonic_cruise"])
    freqs = MICRO_HZ.get(missile_type, MICRO_HZ["supersonic_cruise"])
    control = control or Control()
    rng = rng or np.random.default_rng()
    eff = _effective_aspect(aspect_rad, control.aoa_rad)   # AoA-tilted body aspect (H7)

    prf = 2000.0                                   # slow-time sample rate (Hz)
    win = 2 * H                                    # samples per STFT window -> H bins after half-spectrum
    hop = win // 2
    n_slow = hop * W + win
    t = np.arange(n_slow) / prf
    s = np.zeros(n_slow, dtype=complex)

    for ctr in centers:
        amp = math.sqrt(max(_center_rcs_lin(ctr, eff), 1e-12))   # PO amplitude (sqrt of RCS)
        # bulk radial term (constant here; bulk Doppler lives in the kinematic channel)
        phase = np.zeros(n_slow)
        # Body-rate anticipation: as the airframe pitches to develop AoA (commanded,
        # ahead of the kinematic turn via the actuator lag), EVERY scattering center gets a common
        # pitch-rate micro-Doppler scaled by the (leading) command cue -> visible on the bright body,
        # not just a small fin. This is what lets micro-Doppler anticipate the maneuver.
        phase += 0.45 * control.lead_cue * np.sin(2 * math.pi * 4.0 * t)
        # class-specific micro-motion of this center
        if missile_type == "marv" and ctr.micro == "control":
            fspin = freqs["spin"]; fprec = freqs["precession"]
            phase += 0.6 * np.sin(2 * math.pi * fspin * t) + 0.3 * np.sin(2 * math.pi * fprec * t)
        elif missile_type == "hgv" and ctr.micro in ("control", "cylinder"):
            phase += 0.4 * np.sin(2 * math.pi * freqs["roll"] * t)
        elif missile_type == "ballistic" and ctr.micro in ("fin", "cylinder"):
            phase += 0.5 * np.sin(2 * math.pi * freqs["tumble"] * t)
        elif missile_type == "supersonic_cruise":
            if ctr.micro == "cavity":
                phase += 0.15 * np.sin(2 * math.pi * freqs["engine"] * t)     # steady inlet tone
            if ctr.micro == "control":
                # control-surface flutter whose AMPLITUDE tracks the (leading) deflection cue
                phase += (0.2 + 0.8 * control.lead_cue) * np.sin(2 * math.pi * freqs["control"] * t)
        elif missile_type in ("rv", "decoy_replica") and ctr.micro in ("cylinder", "tip"):
            phase += (0.5 * np.sin(2 * math.pi * freqs["spin"] * t)
                      + 0.25 * np.sin(2 * math.pi * freqs["precession"] * t))   # spin RV / replica mimics it
        elif (missile_type in ("decoy", "bus", "decoy_balloon", "decoy_chaff")
              and ctr.micro in ("fin", "cylinder", "tip", "control")):
            phase += 0.6 * np.sin(2 * math.pi * freqs["tumble"] * t)            # tumble: light decoy/bus/balloon(slow)/chaff(fast)
        # generic maneuver modulation for any class, amplitude = leading cue (anticipatory)
        if ctr.micro in ("control", "fin"):
            phase += 0.5 * control.lead_cue * np.sin(2 * math.pi * 8.0 * t)
        s += amp * np.exp(1j * 2.0 * math.pi * phase)

    # HGV plasma sheath: Doppler broadening (grounded in DeepONet bow-shock) = additive spread.
    if control.plasma > 0.0:
        s += control.plasma * 0.5 * (rng.standard_normal(n_slow) + 1j * rng.standard_normal(n_slow))
    s += 0.03 * noise_scale * (rng.standard_normal(n_slow) + 1j * rng.standard_normal(n_slow))   # noise_scale: SNR knob (1.0 = default), >1 lowers SNR

    spec = np.zeros((H, W), dtype=np.float32)
    for k in range(W):
        seg = s[k * hop: k * hop + win]
        col = _doppler_fft(seg)                    # length `win` (=2H), fftshifted -> zero-Doppler at the centre
        lo = (len(col) - H) // 2                    # keep the CENTRED H bins so the zero-Doppler ridge sits at row H/2
        spec[:, k] = col[lo: lo + H]                #   (was col[:H] = the negative half -> zero-Doppler stuck at the edge)
    m = spec.max()
    return (spec / m).astype(np.float32) if m > 0 else spec


# ---------------------------------------------------------------------------
# Helper: build the per-step Control drivers for a whole trajectory (used by the generator).
# ---------------------------------------------------------------------------
def control_series_from_trajectory(P: np.ndarray, V: np.ndarray, dt: float,
                                   missile_type: str, alts_m: np.ndarray,
                                   cmd_lat_accel: Optional[np.ndarray] = None,
                                   smooth_s: float = 1.0) -> List[Control]:
    """Per-step Control drivers. The anticipation cue (`lead_cue`) is driven by the
    COMMANDED lateral acceleration `cmd_lat_accel` (the control INPUT recorded by the generator),
    NOT by the outcome time-shifted. Because the trajectory applies that command with an actuator
    lag, the command physically precedes the kinematic turn, so the cue leads WITHOUT any circular
    back-dating. If no command is supplied (e.g. a legacy trajectory), the cue falls back to the
    kinematic envelope with ZERO lead (honest: no anticipation without a command channel).
    `aoa_rad` tilts the body axis off the velocity vector ∝ lateral load."""
    n = len(P)
    # KINEMATIC lateral acceleration from velocity (the label / what the tracker sees)
    A = np.zeros_like(V)
    A[1:-1] = (V[2:] - V[:-2]) / (2 * dt)
    lat = np.zeros(n)
    for i in range(n):
        s = np.linalg.norm(V[i])
        if s < 1.0:
            continue
        vhat = V[i] / s
        lat[i] = np.linalg.norm(A[i] - np.dot(A[i], vhat) * vhat)

    k = max(3, int(smooth_s / dt))                      # cue-filter length (s); shorter = less lead suppressed (de-suppression sweep)
    kernel = np.ones(k) / k
    # Cue source: the COMMAND if available (independent input -> physical lead), else kinematic.
    src = np.abs(np.asarray(cmd_lat_accel, dtype=float)) if cmd_lat_accel is not None else lat
    if len(src) != n:                                   # safety: align length
        src = lat
    ref = max(np.percentile(src, 95), 1.0)
    # CAUSAL (trailing) smoothing: the cue at time i must depend only on src[<= i]. A centered
    # convolution (mode="same") looks ~0.5 s into the FUTURE -> on the no-command fallback branch
    # (src = kinematic lateral accel = the maneuver itself) that leaks the future maneuver into the
    # current signature feature. mode="full"[:n] is the backward-only average;
    # on the command branch the command still leads the bend by ACTUATOR_LAG_S, so the anticipation cue
    # is preserved -- now honestly causal on both branches.
    cue = np.clip(np.convolve(np.clip(src / ref, 0.0, 1.0), kernel, mode="full")[:n], 0.0, 1.0)

    # AoA proxy (H7): the body pitches to generate the commanded lift; AoA ~ lateral load / (C_Lα q).
    # Use a bounded map of the kinematic lateral load to a plausible AoA (deg), capped ~20 deg.
    g = 9.80665
    aoa = np.clip(np.degrees(np.arctan2(lat, 3.0 * g)), 0.0, 22.0)   # ~0 cruise, up to ~22 deg in hard turns
    aoa_rad = np.radians(aoa)

    out = []
    for i in range(n):
        plasma = 0.0
        if missile_type == "hgv":
            mach = np.linalg.norm(V[i]) / 343.0
            if mach > 5.0 and alts_m[i] < 60000.0:
                plasma = float(np.clip((mach - 5.0) / 3.0, 0.0, 1.0))
        out.append(Control(lat_accel_mps2=float(lat[i]), lead_cue=float(cue[i]),
                           plasma=plasma, aoa_rad=float(aoa_rad[i])))
    return out
