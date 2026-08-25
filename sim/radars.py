"""
radars.py - Radar Site Configuration and ECEF Conversion

This module defines radar sites in geodetic coordinates and converts them
to ECEF for use in the simulation. It provides:
- Radar site definitions (locations, noise parameters, update rates)
- Geodetic to ECEF coordinate conversion
- Measurement covariance matrices for EKF
- Simple scheduling helpers
"""

import math

# -----------------------
# 1) Human-editable radar site definitions
# -----------------------
RADARS = [
    {
        "id": "dc_east_10km",
        "lat_deg": 38.907189,    # computed: 10km east of DC
        "lon_deg": -76.923629,   # computed
        "alt_m": 0.0,
        "update_rate_hz": 2.0,
        "sigma_range_m": 50.0,
        "sigma_az_rad": 0.0009,   # ~0.05 degrees
        "sigma_el_rad": 0.0009,
        "sigma_doppler_mps": 3.0,
        "dropout_prob": 0.05,
    },
    {
        "id": "nyc_west_10km", 
        "lat_deg": 40.730602,    # computed: 10km west of NYC
        "lon_deg": -74.046727,   # computed
        "alt_m": 0.0,
        "update_rate_hz": 2.0,
        "sigma_range_m": 50.0,
        "sigma_az_rad": 0.0009,
        "sigma_el_rad": 0.0009,
        "sigma_doppler_mps": 3.0,
        "dropout_prob": 0.05,
    },
    {
        "id": "midpoint_west_15km",
        "lat_deg": 39.825905,    # computed: 15km west of NYC-DC midpoint
        "lon_deg": -75.694756,   # computed
        "alt_m": 0.0,
        "update_rate_hz": 2.0,
        "sigma_range_m": 50.0,
        "sigma_az_rad": 0.0009,
        "sigma_el_rad": 0.0009,
        "sigma_doppler_mps": 3.0,
        "dropout_prob": 0.05,
    },
    {
        "id": "pennsylvania_radar",
        "lat_deg": 40.051626,    # supplied coordinate
        "lon_deg": -77.344419,   # supplied coordinate
        "alt_m": 0.0,
        "update_rate_hz": 2.0,
        "sigma_range_m": 50.0,
        "sigma_az_rad": 0.0009,
        "sigma_el_rad": 0.0009,
        "sigma_doppler_mps": 3.0,
        "dropout_prob": 0.05,
    },
]

# --------------------------------
# 2) WGS-84 constants and conversion helpers
# --------------------------------
WGS84_A = 6378137.0                 # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563       # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared

def deg2rad(degrees):
    """Convert degrees to radians"""
    return degrees * math.pi / 180.0

def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """
    Convert geodetic coordinates (WGS-84) to ECEF Cartesian coordinates.
    
    Args:
        lat_deg: latitude in degrees
        lon_deg: longitude in degrees  
        alt_m: altitude above ellipsoid in meters
        
    Returns:
        tuple: (X, Y, Z) in ECEF meters
    """
    φ = deg2rad(lat_deg)
    λ = deg2rad(lon_deg)
    
    sinφ, cosφ = math.sin(φ), math.cos(φ)
    sinλ, cosλ = math.sin(λ), math.cos(λ)
    
    # Prime vertical radius of curvature
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sinφ * sinφ)
    
    # ECEF coordinates
    X = (N + alt_m) * cosφ * cosλ
    Y = (N + alt_m) * cosφ * sinλ
    Z = (N * (1.0 - WGS84_E2) + alt_m) * sinφ
    
    return (X, Y, Z)

def measurement_covariance_diagonal(sigma_range_m, sigma_az_rad, sigma_el_rad, sigma_doppler_mps):
    """
    Create diagonal measurement covariance from noise sigmas.
    
    Args:
        sigma_range_m: range noise standard deviation [m]
        sigma_az_rad: azimuth noise standard deviation [rad]
        sigma_el_rad: elevation noise standard deviation [rad]
        sigma_doppler_mps: doppler noise standard deviation [m/s]
        
    Returns:
        list: diagonal entries [σ_r², σ_az², σ_el², σ_dop²] for R matrix
    """
    return [
        sigma_range_m ** 2,
        sigma_az_rad ** 2,
        sigma_el_rad ** 2,
        sigma_doppler_mps ** 2,
    ]

# -----------------------------------------
# 3) Build math-ready ECEF radar list
# -----------------------------------------
RADARS_ECEF = []

for site in RADARS:
    # Convert geodetic to ECEF once at import time
    x, y, z = geodetic_to_ecef(site["lat_deg"], site["lon_deg"], site["alt_m"])
    
    # Precompute measurement covariance diagonal
    R_diag = measurement_covariance_diagonal(
        site["sigma_range_m"],
        site["sigma_az_rad"], 
        site["sigma_el_rad"],
        site["sigma_doppler_mps"]
    )
    
    # Create enhanced radar entry with ECEF and covariance
    radar_ecef = {
        **site,  # Keep original fields for documentation
        "ecef_m": (x, y, z),     # ECEF position in meters
        "R_diag": R_diag,        # Measurement covariance diagonal
    }
    
    RADARS_ECEF.append(radar_ecef)

# ---------------------------------------------------
# 4) Optional scheduling and utility helpers
# ---------------------------------------------------
def should_radar_emit(prev_time, current_time, update_rate_hz):
    """
    Simple timing check for radar emissions based on update rate.
    
    Args:
        prev_time: previous measurement time
        current_time: current simulation time
        update_rate_hz: radar update rate in Hz
        
    Returns:
        bool: True if enough time has elapsed for next measurement
    """
    if update_rate_hz <= 0:
        return False
    return (current_time - prev_time) >= (1.0 / update_rate_hz)

def get_active_radar_indices(current_time, last_measurement_times):
    """
    Get indices of radars that should emit measurements at current time.
    
    Args:
        current_time: current simulation time
        last_measurement_times: dict mapping radar_id to last measurement time
        
    Returns:
        list: indices of radars that should emit measurements
    """
    active_indices = []
    
    for i, radar in enumerate(RADARS_ECEF):
        radar_id = radar["id"]
        update_rate = radar["update_rate_hz"]
        
        # Get last measurement time (default to 0 if first measurement)
        last_time = last_measurement_times.get(radar_id, 0.0)
        
        if should_radar_emit(last_time, current_time, update_rate):
            active_indices.append(i)
    
    return active_indices

def print_radar_summary():
    """Print a summary of all configured radars"""
    print("\nRadar Site Summary:")
    print("=" * 80)
    for i, radar in enumerate(RADARS_ECEF):
        print(f"Radar {i}: {radar['id']}")
        print(f"  Geodetic: ({radar['lat_deg']:.6f}°, {radar['lon_deg']:.6f}°, {radar['alt_m']:.1f}m)")
        print(f"  ECEF: ({radar['ecef_m'][0]:.1f}, {radar['ecef_m'][1]:.1f}, {radar['ecef_m'][2]:.1f}) m")
        print(f"  Update Rate: {radar['update_rate_hz']:.1f} Hz")
        print(f"  Noise: σ_r={radar['sigma_range_m']:.1f}m, σ_az={radar['sigma_az_rad']:.4f}rad, σ_el={radar['sigma_el_rad']:.4f}rad, σ_dop={radar['sigma_doppler_mps']:.1f}m/s")
        print()

# Export the key items for use by other modules
__all__ = [
    'RADARS',           # Original geodetic definitions
    'RADARS_ECEF',      # Math-ready ECEF definitions
    'geodetic_to_ecef', # Coordinate conversion function
    'should_radar_emit', # Timing helper
    'get_active_radar_indices', # Active radar selector
    'print_radar_summary',      # Debug helper
]

# Print summary when module is imported (optional)
if __name__ == "__main__":
    print_radar_summary()
