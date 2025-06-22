import json
import os

import cv2
import numpy as np

from src import config
from src.evaluation.joint_metrics import compute_ideal_form
from src.utils.helpers import get_pose_bounding_box
from src.visualization.overlay_renderer import draw_angles, draw_feedback_connectors, \
    draw_skeleton_with_colors, render_feedback_panel, add_header_to_frame, render_improved_form_feedback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERMEDIATE_FEEDBACK_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "assets",
                                                   "intermediate_feedback_templates.json")

with open(INTERMEDIATE_FEEDBACK_TEMPLATE_PATH, "r") as f:
    intermediate_feedback_template = json.load(f)


def get_feedback_text(joint_name: str, user_value: float, ideal_value: float, technique: str) -> str:
    """
    Generates intermediate-level feedback for a joint based on deviation from ideal.

    Args:
        joint_name (str): The name of the joint (e.g. "front_knee_angle")
        user_value (float): The computed joint angle from pose data
        ideal_value (float): The ideal angle for the given joint
        technique (str): The technique name (e.g. "cover_drive")

    Returns:
        str: Feedback string with user and ideal values filled in
    """
    if technique not in intermediate_feedback_template:
        return f"No feedback templates available for technique: {technique}"

    if joint_name not in intermediate_feedback_template[technique]:
        return f"No feedback available for joint: {joint_name}"

    # Determine direction of error
    if user_value < ideal_value:
        direction = "too_low"
    else:
        direction = "too_high"

    # Load and format the template
    try:
        template = intermediate_feedback_template[technique][joint_name][direction]
        return template.format(user=round(user_value), ideal=round(ideal_value)).replace("°", "\u00B0")
    except KeyError:
        return f"Feedback template missing for {joint_name} - {direction}"


def generate_feedback(technique, player_frame, worst_joints, keypoints, deviations, user_angles, ideal_pose):
    ideal_frame = player_frame.copy()

    # Compute improved keypoints
    improved_keypoints = compute_ideal_form(
        keypoints,
        ideal_pose,
        user_angles,
        config.ANGLE_COMPONENTS,
        worst_joints
    )

    draw_angles(player_frame, keypoints, deviations, worst_joints, user_opacity=0.5, improved_opacity=0.5)
    draw_angles(ideal_frame, keypoints, deviations, worst_joints, improved_keypoints, config.IDEAL_ANGLE_COLOR,
                user_opacity=0.5, improved_opacity=0.5)

    # Render visual explanation
    # draw_feedback_connectors(player_frame, worst_joints, keypoints)
    # draw_feedback_connectors(ideal_frame, worst_joints, keypoints)

    draw_skeleton_with_colors(player_frame, keypoints, deviations, worst_joints, opacity_lines=0.5,
                              opacity_points=0.5)
    draw_skeleton_with_colors(ideal_frame, keypoints, deviations, worst_joints, opacity_lines=0.5,
                              opacity_points=0.5)
    ideal_frame = render_improved_form_feedback(ideal_frame, improved_keypoints,
                                                   deviations, opacity_lines=0.5, opacity_points=0.5)

    x_min, y_min, x_max, y_max = get_pose_bounding_box(keypoints)
    cv2.rectangle(player_frame, (0, 0), (x_max, y_max), (0, 0, 0), config.BORDER_THICKNESS)
    cv2.rectangle(ideal_frame, (0, 0), (x_max, y_max), (0, 0, 0), config.BORDER_THICKNESS)

    player_frame = add_header_to_frame(player_frame, "Your Pose")
    ideal_frame = add_header_to_frame(ideal_frame, "Ideal Pose")

    # Prepare full canvas
    player_frame_height, player_frame_width, _ = player_frame.shape

    full_width = player_frame_width * 2
    combined_frame = np.ones((player_frame_height, full_width, 3),
                             dtype=np.uint8) * 255  # white background

    # Place the original player frame in left third
    combined_frame[:, :player_frame_width] = player_frame
    combined_frame[:, player_frame_width:] = ideal_frame

    feedback = [get_feedback_text(joint, user_angles[joint], ideal_pose[joint], technique) for joint in
                worst_joints]

    return render_feedback_panel(combined_frame, feedback)
