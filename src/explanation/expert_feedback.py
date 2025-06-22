import json
import os
from typing import List, Dict, Any

import cv2
import numpy as np

from src import config
from src.evaluation.joint_metrics import compute_ideal_form
from src.utils.helpers import get_pose_bounding_box
from src.visualization.overlay_renderer import draw_feedback_connectors, draw_angles, \
    draw_skeleton_with_colors, add_header_to_frame, render_improved_form_feedback

JOINT_CONNECTIONS = [
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
]

JOINT_STRING_REPRESENTATION = {
    "front_knee_angle": "Front Knee Angle",
    "back_knee_angle": "Back Knee Angle",
    "front_elbow_angle": "Front Elbow Angle",
    "back_elbow_angle": "Back Elbow Angle",
    "shoulder_rotation": "Shoulder Rotation",
    "hip_rotation": "Hip Rotation",
    "spine_forward_lean": "Spine Forward Lean",
    "head_tilt": "Head Tilt",
    "bat_angle": "Bat Angle"
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERT_FEEDBACK_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "assets",
                                             "expert_feedback_templates.json")

SHAP_VALUES_PATH = os.path.join(PROJECT_ROOT, "assets", "shap_inputs.json")

with open(EXPERT_FEEDBACK_TEMPLATE_PATH, "r") as f:
    expert_feedback_template = json.load(f)

with open(SHAP_VALUES_PATH, "r") as f:
    shap_inputs = json.load(f)


def get_feedback_text(joint_name: str, user_value: float, ideal_value: float, technique: str,
                      shap_value: float) -> str:
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
    if technique not in expert_feedback_template:
        return f"No feedback templates available for technique: {technique}"

    if joint_name not in expert_feedback_template[technique]:
        return f"No feedback available for joint: {joint_name}"

    # Determine direction of error
    if user_value < ideal_value:
        direction = "too_low"
    else:
        direction = "too_high"

    # Load and format the template
    try:
        template = expert_feedback_template[technique][joint_name][direction]
        return template.format(user=round(user_value), ideal=round(ideal_value),
                               shap_value=round(shap_value * 100)).replace("°", "\u00B0")
    except KeyError:
        return f"Feedback template missing for {joint_name} - {direction}"


def draw_feedback_table(
        frame: np.ndarray,
        feedback_rows: List[Dict[str, Any]],
        base_row_height: int = 30,
        padding: int = 5,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
        thickness: int = 1,
        header_color: tuple = (200, 200, 200),
        line_color: tuple = (0, 0, 0),
        text_color: tuple = (0, 0, 0),
        max_deviation: float = 100.0,  # fixed maximum deviation for color scaling
) -> np.ndarray:
    h, w = frame.shape[:2]
    table_w = w

    cols = ["Joint", "Your Angle", "Ideal Angle", "Deviation", "SHAP (%)", "Suggestion"]
    keys = ["joint", "user", "ideal", "deviation", "shap", "suggestion"]

    # measure text size
    def text_size(txt):
        (tw, th), _ = cv2.getTextSize(str(txt), font, font_scale, thickness)
        return tw, th

    # 1) compute column widths
    col_widths = []
    for i, col in enumerate(cols[:-1]):
        max_w = text_size(col)[0] + 2 * padding
        for row in feedback_rows:
            max_w = max(max_w, text_size(row.get(keys[i], ""))[0] + 2 * padding)
        col_widths.append(max_w)

    used = sum(col_widths)
    rem = table_w - used - 2 * padding
    col_widths.append(max(rem, text_size(cols[-1])[0] + 2 * padding))

    # 2) wrap suggestions and compute row heights
    wrapped_suggestions: List[List[str]] = []
    line_h = text_size("Ay")[1] + 2
    row_heights = []
    for row in feedback_rows:
        words = str(row.get("suggestion", "")).split()
        lines, cur = [], ""
        max_w = col_widths[-1] - 2 * padding
        for w0 in words:
            test = (cur + " " + w0).strip()
            if text_size(test)[0] <= max_w:
                cur = test
            else:
                lines.append(cur)
                cur = w0
        if cur:
            lines.append(cur)
        wrapped_suggestions.append(lines)
        height = max(base_row_height, len(lines) * line_h + 2 * padding)
        row_heights.append(height)

    # 3) compute table vertical centering
    header_h = base_row_height
    total_h = header_h + sum(row_heights)
    y0 = max((h - total_h) // 2, 0)

    # 4) create white canvas for table
    table_frame = np.ones((h, table_w, 3), dtype=np.uint8) * 255

    # 5) draw header row
    x = 0
    for i, hdr in enumerate(cols):
        cw = col_widths[i]
        cv2.rectangle(table_frame, (x, y0), (x + cw, y0 + header_h), header_color, -1)
        cv2.rectangle(table_frame, (x, y0), (x + cw, y0 + header_h), line_color, 1)
        cv2.putText(table_frame, hdr,
                    (x + padding, y0 + header_h - padding),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)
        x += cw

    # 6) use fixed max_deviation for color scaling
    max_dev = max_deviation or 1.0

    # 7) draw data rows with gradient based on fixed max_deviation
    y = y0 + header_h
    for ridx, row in enumerate(feedback_rows):
        rh = row_heights[ridx]
        lines = wrapped_suggestions[ridx]
        x = 0
        for i, key in enumerate(keys):
            cw = col_widths[i]
            # gradient coloring for deviation magnitude
            if key == "deviation":
                d = abs(float(row.get("deviation", 0)))
                norm = min(d / max_dev, 1.0)
                if norm < 0.5:
                    # interpolate from green (at zero) to yellow (mid)
                    r = int((norm / 0.5) * 255)
                    g = 255
                else:
                    # interpolate from yellow (mid) to red (max)
                    r = 255
                    g = int((1 - (norm - 0.5) / 0.5) * 255)
                b = 0
                bg = (b, g, r)
                cv2.rectangle(table_frame, (x, y), (x + cw, y + rh), bg, -1)
            # cell border
            cv2.rectangle(table_frame, (x, y), (x + cw, y + rh), line_color, 1)
            # render text
            if key == "suggestion":
                for li, txt in enumerate(lines):
                    ty = y + padding + (li + 1) * line_h - (line_h // 4)
                    cv2.putText(table_frame, txt,
                                (x + padding, ty),
                                font, font_scale, text_color, thickness, cv2.LINE_AA)
            else:
                txt = str(row.get(key, ""))
                ty = y + rh - padding
                cv2.putText(table_frame, txt,
                            (x + padding, ty),
                            font, font_scale, text_color, thickness, cv2.LINE_AA)
            x += cw
        y += rh

    # 8) merge with original frame
    combined = np.ones((h, w * 2, 3), dtype=np.uint8) * 255
    combined[:, :w] = frame
    combined[:, w:] = table_frame
    return combined


def generate_feedback(technique, player_frame, worst_joints: dict, keypoints, deviations, user_angles,
                      ideal_pose):
    ideal_frame = player_frame.copy()

    # Compute improved keypoints
    improved_keypoints = compute_ideal_form(
        keypoints,
        ideal_pose,
        user_angles,
        config.ANGLE_COMPONENTS,
        worst_joints
    )

    draw_feedback_connectors(player_frame, worst_joints, keypoints)
    draw_feedback_connectors(ideal_frame, worst_joints, keypoints)

    draw_angles(player_frame, keypoints, deviations, worst_joints, user_opacity=0.5, improved_opacity=0.5)
    draw_angles(ideal_frame, keypoints, deviations, worst_joints, improved_keypoints, config.IDEAL_ANGLE_COLOR,
                user_opacity=0.5, improved_opacity=0.5)

    # Render visual explanation

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

    shap_values = shap_inputs[technique]

    feedback_rows = []
    max_widths = [0, 0, 0, 0, 0, 0]
    for i, joint in enumerate(worst_joints.keys()):
        deviation = round(deviations[joint], 2)
        user = round(user_angles[joint])
        ideal = round(ideal_pose[joint])
        shap = round(shap_values[joint] * 100)

        direction = "too_high" if user > ideal else "too_low"
        feedback = expert_feedback_template[technique][joint][direction]

        row = {
            "joint": JOINT_STRING_REPRESENTATION[joint],
            "user": user,
            "ideal": ideal,
            "shap": shap,
            "deviation": deviation,
            "suggestion": feedback,
        }

        for i, x in enumerate(row.values()):
            (tw, _), _ = cv2.getTextSize(str(x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_widths[i] = max(max_widths[i], tw)
            pass

        feedback_rows.append(row)

    feedback_rows.reverse()
    return draw_feedback_table(combined_frame, feedback_rows)
