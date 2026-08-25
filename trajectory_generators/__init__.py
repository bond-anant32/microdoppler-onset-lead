# trajectory_generators package
# Force-integrated and kinematic truth-trajectory generators. Every generator
# returns the same contract used elsewhere in the project:
#     positions : (N, 3) float ndarray, ECEF metres
#     dt        : float, seconds between samples
# so the measurement / EKF / ML layers consume them unchanged.
