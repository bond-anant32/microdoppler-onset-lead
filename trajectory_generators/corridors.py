"""
trajectory_generators/corridors.py  --  Phase 1 of the sensor-architecture build.

Trajectory-FOLLOWS-scenario (owner rule). Instead of random global endpoints with radars
dropped along the ground track (which placed "ground radars" on the open ocean -- the
project-state "radar floating on the ocean" defect), we define a small set of fixed, realistic
THREAT CORRIDORS. Each corridor is {launch region on real land, defended region on real land,
allowed missile classes, and a plausible LAND-ONLY sensor lay-down}. Trajectories are generated
launch -> defended-region, and the sensors are placed at plausible land sites positioned to
observe that corridor -- so custody handover from the launch-side network to the defended-side
network falls out of the geometry (a descending target drops below the launch-side horizon into
the defended-side radar's line of sight, via the existing has_line_of_sight() gate).

Land is enforced with `global_land_mask.globe.is_land` (verified: DC/Moscow/Tehran -> land,
mid-Atlantic/Pacific -> ocean). Sensors that would land on ocean are nudged to the nearest land;
if none is within reach (deep-ocean glide leg), NO ground sensor is placed there -- that is
exactly the over-horizon gap the limited satellite layer (Phase 5) is designed to fill.

Sensor tiers (Phase 1 tags band/freq; Phase 2 makes VHF anti-stealth RCS real):
  land_x  : X-band fire-control (precise, short-medium range, fast update)
  land_s  : S-band acquisition
  vhf_ew  : VHF Voronezh/Nebo-class early-warning (coarse, very long range, slow) -- anti-stealth
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from global_land_mask import globe
from trajectory_generators.geo import dest_point
from trajectory_math import geodetic_to_ecef


# Strategic islands / sites the 0.1-deg land mask is too coarse to resolve but which host real
# radars (Guam AN/TPY-2, Okinawa, Taiwan, Hawaii, ...). Treated as land within ~60 km.
KNOWN_SITES = [
    (13.44, 144.79, "Guam"), (26.34, 127.80, "Okinawa"), (23.70, 121.00, "Taiwan"),
    (21.30, -157.86, "Hawaii"), (7.40, 134.50, "Palau"), (28.20, 129.30, "Amami"),
]


def is_land(lat: float, lon: float) -> bool:
    """Raw land mask only (ocean-rejection)."""
    return bool(globe.is_land(float(lat), float(((lon + 180.0) % 360.0) - 180.0)))


def is_land_or_site(lat: float, lon: float, site_r_km: float = 60.0) -> bool:
    """Land per the mask OR within site_r_km of a known strategic site (small-island override)."""
    if is_land(lat, lon):
        return True
    return any(_great_circle_km((lat, lon), (sla, slo)) <= site_r_km for sla, slo, _ in KNOWN_SITES)


def nearest_land(lat: float, lon: float, max_r_km: float = 350.0, step_km: float = 50.0):
    """Spiral outward to the nearest land/site point; None if all ocean within max_r_km."""
    if is_land_or_site(lat, lon):
        return (lat, lon)
    r = step_km
    while r <= max_r_km:
        for b in range(0, 360, 30):
            la, lo = dest_point(lat, lon, b, r)
            if is_land_or_site(la, lo):
                return (la, lo)
        r += step_km
    return None


def _great_circle_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    d = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(d))


def _bearing(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# Sensor presets (the dict contract simulate_radar_measurements already consumes, + band/freq/kind).
_PRESETS = {
    "land_x": dict(freq=9.0e9, band="X",   rate=3.0, sr=50.0,  sa=0.0009, sd=3.0, pw=112.0),
    "land_s": dict(freq=3.3e9, band="S",   rate=2.0, sr=70.0,  sa=0.0015, sd=4.0, pw=113.0),
    "vhf_ew": dict(freq=1.5e8, band="VHF", rate=0.15, sr=500.0, sa=0.017,  sd=10.0, pw=118.0),
}


def make_sensor(sid: str, lat: float, lon: float, alt_m: float, kind: str) -> Dict:
    p = _PRESETS[kind]
    x, y, z = geodetic_to_ecef(lat, lon, alt_m)
    return {
        "id": sid, "lat_deg": lat, "lon_deg": lon, "alt_m": alt_m,
        "ecef_m": (float(x), float(y), float(z)),
        "update_rate_hz": p["rate"], "sigma_range_m": p["sr"],
        "sigma_az_rad": p["sa"], "sigma_el_rad": p["sa"], "sigma_doppler_mps": p["sd"],
        "R_diag": [p["sr"] ** 2, p["sa"] ** 2, p["sa"] ** 2, p["sd"] ** 2],
        "power_db": p["pw"], "band": p["band"], "freq_hz": p["freq"], "sensor_kind": kind,
    }


@dataclass
class Corridor:
    """A fixed threat geometry: launch region + defended region (both on land) + sensor lay-down."""
    name: str
    launch_center: Tuple[float, float]     # (lat, lon) on land
    target_center: Tuple[float, float]     # (lat, lon) on land
    spread_deg: float                      # launch/target sampling box half-width
    classes: Tuple[str, ...]               # allowed missile types
    transoceanic: bool = False             # glide/cruise leg crosses ocean -> satellite needed

    def sample_endpoints(self, rng) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Rejection-sample a launch and target that are BOTH on land (centroid fallback)."""
        def _samp(center):
            for _ in range(50):
                la = center[0] + float(rng.uniform(-self.spread_deg, self.spread_deg))
                lo = center[1] + float(rng.uniform(-self.spread_deg, self.spread_deg))
                if is_land_or_site(la, lo):
                    return (la, lo)
            return center                          # trusted corridor site (hand-chosen real location)
        return _samp(self.launch_center), _samp(self.target_center)

    def sensors(self, rng, launch_ll, target_ll) -> List[Dict]:
        """Place a plausible LAND-ONLY sensor lay-down for this corridor. A VHF EW site stands
        deep behind the launch (anti-stealch cue); X-band fire-control near launch and near the
        defended asset. Sites over ocean are nudged to land or dropped (satellite fills the gap)."""
        brg = _bearing(launch_ll, target_ll)
        out: List[Dict] = []

        def _add(kind, lat, lon, tag):
            land = nearest_land(lat, lon)
            if land is not None:
                out.append(make_sensor(f"{kind}_{tag}", land[0], land[1], 0.0, kind))

        # VHF early-warning: deep standoff behind the launch (long-range anti-stealth cue).
        ew = dest_point(*launch_ll, (brg + 180.0) % 360.0, float(rng.uniform(220.0, 400.0)))
        _add("vhf_ew", ew[0], ew[1], "0")
        # Launch-side X-band fire-control (acquire the boost/midcourse). Flank angle + range are jittered per
        # site so the viewing geometry (aspect) varies. Held OFF the TEL (>=120 km) so the sensor markers do
        # not visually MERGE with the launch point on the globe (a radar is not co-located with the TEL).
        for k, side in enumerate((55.0, -55.0)):
            ang = side + float(rng.uniform(-25.0, 25.0))
            ls = dest_point(*launch_ll, (brg + ang) % 360.0, float(rng.uniform(120.0, 260.0)))
            _add("land_x", ls[0], ls[1], f"L{k}")
        # Defended-side X-band fire-control + S-band acquisition form a DEFENSIVE RING around the asset --
        # standoff (>=60 km) from the defended point, NOT sitting on the impact/interceptor marker (the missile
        # is aimed at the asset, the radars defend it from around it -- they are not the target).
        for k, side in enumerate((40.0, -40.0)):
            ang = side + float(rng.uniform(-30.0, 30.0))
            ts = dest_point(*target_ll, (brg + 180.0 + ang) % 360.0, float(rng.uniform(70.0, 200.0)))
            _add("land_x", ts[0], ts[1], f"T{k}")
        # S-band on a FLANK of the asset (not on the threat's approach axis brg+180, where the incoming missile
        # flies right over it -- the "missile strikes the radar" look), so the defensive ring surrounds the asset
        # without any sensor sitting under the impact path.
        sc = dest_point(*target_ll, (brg + 180.0 + 70.0) % 360.0, float(rng.uniform(60.0, 110.0)))
        _add("land_s", sc[0], sc[1], "TC")
        return out


# The 6 corridors (the right count, not a 55-lever farm). Real land regions; ocean-crossing
# corridors are flagged transoceanic so Phase 5 phases a satellite over their glide leg.
CORRIDORS: List[Corridor] = [
    Corridor("C1_iran_israel",   (32.0, 53.0),  (31.6, 35.0),  2.5, ("ballistic", "marv", "mirv")),
    Corridor("C2_nk_kyushu",     (40.0, 127.0), (32.8, 130.6), 2.0, ("ballistic", "marv", "mirv"), transoceanic=True),
    Corridor("C3_cont_vhf",      (50.0, 40.0),  (55.0, 25.0),  3.0, ("supersonic_cruise", "hgv")),
    Corridor("C4_china_guam",    (25.0, 118.0), (13.44, 144.79), 1.0, ("hgv", "ballistic"), transoceanic=True),
    Corridor("C5_coastal_seaskim", (36.0, 138.5), (35.5, 139.7), 1.2, ("supersonic_cruise",)),
    Corridor("C6_highloft_bal",  (45.0, 60.0),  (52.0, 40.0),  3.0, ("ballistic",)),
    # long-range transcontinental MIRV/ICBM shot (~2600 km over land, below the bus reachability cap so the RVs
    # land on target and defended-side radars keep custody) -- the regional C1/C2 hops were too short for a MIRV.
    Corridor("C7_transcon_mirv", (47.0, 80.0),  (49.0, 45.0),  3.0, ("mirv", "ballistic", "marv", "hgv", "supersonic_cruise")),
    # even longer intercontinental land shots (~3500-4000 km): only the classes that PHYSICALLY range this far --
    # ballistic/MIRV (ICBM), boost-glide HGV, hypersonic cruise. A MaRV is a MEDIUM-range sub-ballistic RV: on a
    # 3800 km shot it flies near-ballistic (apogee ~950 km, no longer discriminably depressed), so it stays on the
    # <=2600 km corridors (C1/C2/C7) where its depressed maneuvering arc is physical.
    Corridor("C8_transcon_long", (55.0, 95.0),  (48.0, 40.0),  3.0, ("ballistic", "hgv", "mirv", "supersonic_cruise")),
]


def corridors_for(missile_type: str) -> List[Corridor]:
    return [c for c in CORRIDORS if missile_type in c.classes]


def random_long_corridor(rng, classes, min_km: float = 1900.0, max_km: float = 3100.0):
    """A RANDOM long land-to-land corridor (owner: don't cluster every scene on NK->SK / Iran->Israel; make them
    all LONG so the engagement is visible + gives a large shoot-look-shoot window). Both endpoints on real land,
    min_km..max_km apart, random location + azimuth. The land-only sensor lay-down still falls out of
    Corridor.sensors, so radars never float on the ocean. Retries until it finds a land-to-land long pair."""
    for _ in range(600):
        la0 = float(rng.uniform(18.0, 60.0)); lo0 = float(rng.uniform(-165.0, 170.0))
        if not is_land(la0, lo0):
            continue
        brg = float(rng.uniform(0.0, 360.0)); span = float(rng.uniform(min_km, max_km))
        la1, lo1 = dest_point(la0, lo0, brg, span)
        tgt = nearest_land(la1, lo1, max_r_km=300.0)                 # nudge an ocean endpoint to the nearest coast
        if tgt is None:
            continue
        d = _great_circle_km((la0, lo0), tgt)
        if not (min_km * 0.85 <= d <= max_km * 1.2):                 # stay long after the land-nudge
            continue
        return Corridor(f"rand_{int(la0)}_{int(lo0)}_{int(brg)}", (la0, lo0), tgt, 1.0, tuple(classes))
    return Corridor("rand_fallback", (55.0, 95.0), (48.0, 40.0), 1.2, tuple(classes))  # a long fixed fallback
