import numpy as np


def normalize_angle(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi
