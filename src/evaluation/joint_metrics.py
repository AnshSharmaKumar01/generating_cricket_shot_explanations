import math
from typing import Dict, Tuple

from src.utils.helpers import calculate_angle


def calculate_joint_angles(keypoints: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Given pose keypoints, compute key angles for the Cover Drive shot.
    """
    angles = {}

    # Knee angles
    angles["front_knee_angle"] = calculate_angle(
        keypoints["LEFT_HIP"], keypoints["LEFT_KNEE"], keypoints["LEFT_ANKLE"]
    )
    angles["back_knee_angle"] = calculate_angle(
        keypoints["RIGHT_HIP"], keypoints["RIGHT_KNEE"], keypoints["RIGHT_ANKLE"]
    )

    # Elbow angles
    angles["front_elbow_angle"] = calculate_angle(
        keypoints["LEFT_SHOULDER"], keypoints["LEFT_ELBOW"], keypoints["LEFT_WRIST"]
    )
    angles["back_elbow_angle"] = calculate_angle(
        keypoints["RIGHT_SHOULDER"], keypoints["RIGHT_ELBOW"], keypoints["RIGHT_WRIST"]
    )

    # Shoulder rotation = angle between shoulders and horizontal
    left_shoulder = keypoints["LEFT_SHOULDER"]
    right_shoulder = keypoints["RIGHT_SHOULDER"]
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angles["shoulder_rotation"] = abs(math.degrees(math.atan2(dy, dx)))

    # Hip rotation = angle between hips and horizontal
    left_hip = keypoints["LEFT_HIP"]
    right_hip = keypoints["RIGHT_HIP"]
    dx = right_hip[0] - left_hip[0]
    dy = right_hip[1] - left_hip[1]
    angles["hip_rotation"] = abs(math.degrees(math.atan2(dy, dx)))

    # Spine lean = angle between shoulder-hip line and vertical
    mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2)
    mid_hip = ((left_hip[0] + right_hip[0]) / 2,
               (left_hip[1] + right_hip[1]) / 2)
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]
    spine_angle = math.degrees(math.atan2(dx, dy))  # angle from vertical
    angles["spine_forward_lean"] = abs(spine_angle)

    # Head tilt = angle between ears
    left_ear = keypoints.get("LEFT_EAR", left_shoulder)  # fallback if ear not detected
    right_ear = keypoints.get("RIGHT_EAR", right_shoulder)
    dx = right_ear[0] - left_ear[0]
    dy = right_ear[1] - left_ear[1]
    angles["head_tilt"] = abs(math.degrees(math.atan2(dy, dx)))

    # Bat angle — estimated from wrist to bat tip (simplified assumption)
    # Assuming bat is held in right hand, and bat tip is approximated by thumb
    if "RIGHT_THUMB" in keypoints:
        bat_vector = (keypoints["RIGHT_THUMB"][0] - keypoints["RIGHT_WRIST"][0],
                      keypoints["RIGHT_THUMB"][1] - keypoints["RIGHT_WRIST"][1])
        angles["bat_angle"] = abs(math.degrees(math.atan2(bat_vector[1], bat_vector[0])))
    else:
        angles["bat_angle"] = 0.0  # fallback if thumb not available

    return angles


def compute_joint_deviation(user_angles: Dict[str, float], ideal_angles: Dict[str, float]) -> Dict[str, float]:
    """
    Computes absolute deviation for each joint angle compared to the ideal.
    Returns: {joint_name: deviation}
    """
    deviations = {}

    angles = user_angles.keys()
    for angle in angles:
        ideal = ideal_angles.get(angle)
        user = user_angles.get(angle)
        deviation = user - ideal
        deviations[angle] = deviation

    return deviations


def rotate_point_around_origin(pivot: Tuple[int, int], point: Tuple[int, int], angle_deg: float) -> Tuple[
    int, int]:
    """Rotate a point around a pivot by a given angle in degrees."""
    angle_rad = math.radians(angle_deg)
    ox, oy = pivot
    px, py = point

    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    return int(qx), int(qy)


def compute_ideal_form(keypoints: Dict[str, Tuple[int, int]],
                       ideal_angles: Dict[str, float],
                       user_angles: Dict[str, float],
                       angle_components: Dict[str, Tuple[str, str, str]],
                       worst_joints: Dict[str, float]) -> Dict[str, Tuple[int, int]]:
    """
    Returns a modified set of keypoints where each worst joint angle is adjusted toward the ideal angle.
    """
    improved = keypoints.copy()

    for aspect in worst_joints:
        if aspect not in angle_components or aspect not in ideal_angles or aspect not in user_angles:
            continue

        joint1, joint2, joint3 = angle_components[aspect]
        joint1 = joint1.upper()
        joint2 = joint2.upper()
        joint3 = joint3.upper()

        if joint1 not in keypoints or joint2 not in keypoints or joint3 not in keypoints:
            continue

        a = keypoints[joint1]
        b = keypoints[joint2]  # central joint
        c = keypoints[joint3]

        current_angle = user_angles[aspect]
        target_angle = ideal_angles[aspect]

        # Calculate how much to rotate c around b
        angle_diff = target_angle - current_angle

        # Rotate point c around point b
        new_c = rotate_point_around_origin(b, c, angle_diff)
        improved[joint3] = new_c
    return improved



if __name__ == '__main__':
    from src.pose_estimation.mediapipe_pose import extract_keypoints
    from src import config
    from src.evaluation.ideal_pose_profiles import get_ideal_pose
    import os

    sample_video_path = config.SAMPLE_VIDEO_PATH
    sample_target_frame = config.SAMPLE_TARGET_FRAME
    sample_technique = config.TECHNIQUE
    sample_frame, sample_keypoints = extract_keypoints(sample_video_path, sample_target_frame)

    sample_user_angles = calculate_joint_angles(sample_keypoints)

    print("######################## USER ANGLES ########################")
    print(sample_user_angles)

    perfect_video_path = config.PERFECT_VIDEO_PATH
    perfect_target_frame = config.PERFECT_TARGET_FRAME
    perfect_technique = config.TECHNIQUE
    perfect_frame, perfect_keypoints = extract_keypoints(perfect_video_path, perfect_target_frame)

    perfect_user_angles = calculate_joint_angles(perfect_keypoints)

    print("######################## PERFECT ANGLES ########################")
    print(perfect_user_angles)
    print("######################## IDEAL ANGLES ########################")
    ideal_angles = get_ideal_pose(perfect_technique, root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    print(ideal_angles)
    print("######################## USER DEVIATIONS ########################")
    print(compute_joint_deviation(sample_user_angles, ideal_angles))
    print("######################## PERFECT DEVIATIONS ########################")
    print(compute_joint_deviation(perfect_user_angles, ideal_angles))
