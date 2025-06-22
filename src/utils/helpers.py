# Helper to compute angle between three points
import math
from typing import Tuple, Dict


def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Computes the angle at point `b` formed by points a-b-c (in degrees)
    """
    def vector(p1, p2):
        return [p2[0] - p1[0], p2[1] - p1[1]]

    ba = vector(b, a)
    bc = vector(b, c)

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba * mag_bc == 0:
        return 0.0  # Prevent division by zero

    angle = math.acos(dot_product / (mag_ba * mag_bc))
    return math.degrees(angle)


def get_pose_bounding_box(keypoints: Dict[str, Tuple[int, int]], padding: int = 100):
    """
    Calculate bounding box that contains all visible keypoints.
    Args:
        keypoints: Dictionary of joint_name -> (x, y)
        padding: Extra pixels to include around the person
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    xs = [pt[0] for pt in keypoints.values()]
    ys = [pt[1] for pt in keypoints.values()]

    x_min = max(min(xs) - padding, 0)
    y_min = max(min(ys) - padding, 0)
    x_max = max(xs) + padding
    y_max = max(ys) + padding

    return int(x_min), int(y_min), int(x_max), int(y_max) + 55
