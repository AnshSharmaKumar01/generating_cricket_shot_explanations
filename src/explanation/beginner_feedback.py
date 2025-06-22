from typing import Dict, List

import json
import os
import cv2

from src import config
from src.utils.helpers import get_pose_bounding_box
from src.visualization.overlay_renderer import draw_feedback_connectors, draw_skeleton_with_colors, \
    render_feedback_panel, add_header_to_frame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEGINNER_FEEDBACK_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "assets", "beginner_feedback_templates.json")
LOW_THRESHOLD = config.LOW_THRESHOLD
MEDIUM_THRESHOLD = config.MEDIUM_THRESHOLD

with open(BEGINNER_FEEDBACK_TEMPLATE_PATH, "r") as f:
    beginner_feedback_templates = json.load(f)


def rank_worst_joints(deviations: Dict[str, float], keypoints: Dict[str, tuple[int, int]], top_k: int = 3) -> \
        dict[str, float]:
    """
    Sort joints by deviation and return the worst K joints.
    """
    sorted_dict = dict(sorted(deviations.items(), key=lambda item: abs(item[1]))[-(top_k):])
    return sorted_dict


def get_magnitude_category(delta: float) -> str:
    """
    Classify the magnitude of deviation into categories.
    """
    if delta < LOW_THRESHOLD:
        return "small"
    elif delta < MEDIUM_THRESHOLD:
        return "medium"
    else:
        return "large"


def get_feedback_text(joint_name: str, deviation: float, technique: str) -> str:
    """
    Returns the feedback text based on how far the user value deviates from the ideal.

    Args:
        joint_name (str): e.g., 'front_knee_angle'
        user_value (float): computed from pose
        ideal_value (float): expected from ideal pose
        technique (str): e.g., 'cover_drive'

    Returns:
        str: explanation for the joint deviation
    """
    if technique not in beginner_feedback_templates:
        return f"No feedback available for technique: {technique}"

    joint_feedback = beginner_feedback_templates[technique].get(joint_name)
    if not joint_feedback:
        return f"No feedback template for joint: {joint_name}"

    magnitude = get_magnitude_category(abs(deviation))

    if deviation < 0:
        direction = "too_low"
    else:
        direction = "too_high"

    try:
        return joint_feedback[direction][magnitude]
    except KeyError:
        return f"No feedback for {joint_name} with {direction} ({magnitude} deviation)."


def generate_feedback(technique, player_frame, worst_joints, keypoints, deviations):
    feedback = [get_feedback_text(joint, deviations[joint], technique) for joint in worst_joints]

    draw_feedback_connectors(player_frame, worst_joints, keypoints)
    draw_skeleton_with_colors(player_frame, keypoints, deviations, worst_joints, opacity_lines=0.5,
                              opacity_points=0.5)

    x_min, y_min, x_max, y_max = get_pose_bounding_box(keypoints)
    cv2.rectangle(player_frame, (0, 0), (x_max, y_max), (0, 0, 0), config.BORDER_THICKNESS)

    player_frame = add_header_to_frame(player_frame, "Your Pose")

    return render_feedback_panel(player_frame, feedback)


if __name__ == '__main__':
    my_dict = {
        'a': -10,
        'b': 3,
        'c': -7,
        'd': 1
    }

    print(rank_worst_joints(my_dict))

    print("##################################################################################################")

    joint_name = "front_knee_angle"
    devition = -50
    technique = "cover_drive"

    print(get_feedback_text(joint_name, devition, technique))
