import math

import cv2
import numpy as np
from typing import List, Dict, Tuple
from src import config

# Define color constants (BGR)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

LOW_THRESHOLD = config.LOW_THRESHOLD
MEDIUM_THRESHOLD = config.MEDIUM_THRESHOLD

# Define skeleton connections
JOINT_CONNECTIONS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip")
]


# Define color thresholds
def get_color_for_deviation(deviation: float, max_deviation: float = 30.0) -> Tuple[int, int, int]:
    """
    Maps deviation to a color from green (low) to red (high) using a gradient.

    Args:
        deviation (float): The deviation magnitude.
        max_deviation (float): The upper bound for deviation normalization.

    Returns:
        Tuple[int, int, int]: BGR color for OpenCV
    """
    # Normalize deviation between 0 (good) and 1 (bad)
    norm = min(deviation / max_deviation, 1.0)

    # Interpolate between green (good) and red (bad)
    red = int(255 * norm)
    green = int(255 * (1 - norm))
    blue = 0

    return (blue, green, red)  # OpenCV uses BGR


def draw_skeleton_with_colors(frame,
                              keypoints: Dict[str, Tuple[int, int]],
                              joint_deviations: Dict[str, float],
                              worst_joints: Dict[str, float],
                              opacity_lines: float = 1.0,
                              opacity_points: float = 1.0):
    """
    Draws a skeleton overlay with adjustable opacity for lines and points.
    """
    overlay_lines = frame.copy()
    overlay_points = frame.copy()

    # Draw skeleton lines on line overlay
    for j1, j2 in JOINT_CONNECTIONS:
        coords1 = tuple(map(int, keypoints[j1.upper()]))
        coords2 = tuple(map(int, keypoints[j2.upper()]))
        cv2.line(overlay_lines, coords1, coords2, config.LINE_COLOR, 2)

    # Prepare joint deviation data
    joint_overlay = {}
    for aspect, deviation in joint_deviations.items():
        for j in config.ANGLE_TO_JOINT.get(aspect, []):
            joint_overlay[j.upper()] = max(joint_overlay.get(j.upper(), 0.0), deviation)

    # Draw focus joints on point overlay
    for joint, deviation in joint_overlay.items():
        coords = keypoints.get(joint.upper())
        if coords:
            cv2.circle(overlay_points, tuple(map(int, coords)), 8, config.FOCUS_JOINT_COLOR, -1)

    # Draw worst joints on point overlay
    for aspect in worst_joints:
        for j in config.ANGLE_TO_JOINT.get(aspect, []):
            coords = keypoints.get(j.upper())
            if coords:
                cv2.circle(overlay_points, tuple(map(int, coords)), 8, config.WORST_JOINT_COLOR, 5)

    # Blend overlays with opacity
    cv2.addWeighted(overlay_lines, opacity_lines, frame, 1 - opacity_lines, 0, frame)
    cv2.addWeighted(overlay_points, opacity_points, frame, 1 - opacity_points, 0, frame)


def draw_angles(frame,
                keypoints: Dict[str, Tuple[int, int]],
                joint_deviations: Dict[str, float],
                worst_joints: Dict[str, float],
                improved_keypoints: Dict[str, Tuple[int, int]] = None,
                angle_color=None,
                user_opacity: float = 1.0,
                improved_opacity: float = 1.0):
    """
    Draws filled angle wedges for both the user and optionally improved poses, with opacity support.
    """
    ANGLE_COMPONENTS = config.ANGLE_TO_JOINT

    def angle_between(p1, p2):
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

    def draw_angle_sector_on(overlay, center, limb1, limb2, color, radius, flip=False):
        start_angle = angle_between(center, limb1) % 360
        end_angle = angle_between(center, limb2) % 360
        angle_diff = (end_angle - start_angle + 360) % 360

        # Always draw the smaller wedge
        if angle_diff > 180 or flip:
            flip = True
            start_angle, end_angle = end_angle, start_angle
            angle_diff = (end_angle - start_angle + 360) % 360

        points = [center]
        for i in range(0, int(angle_diff) + 1, 2):
            angle_deg = (start_angle + i) % 360
            rad = math.radians(angle_deg)
            x = int(center[0] + radius * math.cos(rad))
            y = int(center[1] + radius * math.sin(rad))
            points.append((x, y))

        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 0), thickness=2)  # black border
        return flip

    # Create overlays
    user_overlay = frame.copy()
    improved_overlay = frame.copy()

    for aspect in worst_joints:
        if aspect not in ANGLE_COMPONENTS:
            continue

        joints = ANGLE_COMPONENTS[aspect]
        if len(joints) != 3:
            continue

        j1, j2, j3 = [j.upper() for j in joints]
        if j1 not in keypoints or j2 not in keypoints or j3 not in keypoints:
            continue

        a, b, c = keypoints[j1], keypoints[j2], keypoints[j3]
        deviation_color = get_color_for_deviation(joint_deviations.get(aspect, 0))
        flip = draw_angle_sector_on(user_overlay, b, a, c, deviation_color, radius=40)

        # Optionally draw improved angle
        if improved_keypoints and all(j in improved_keypoints for j in (j1, j2, j3)):
            a2, b2, c2 = improved_keypoints[j1], improved_keypoints[j2], improved_keypoints[j3]
            draw_color = angle_color if angle_color else (0, 255, 0)
            draw_angle_sector_on(improved_overlay, b2, a2, c2, draw_color, radius=30, flip=flip)

    # Blend overlays with opacity
    if user_opacity < 1.0:
        cv2.addWeighted(user_overlay, user_opacity, frame, 1 - user_opacity, 0, frame)
    else:
        frame[:] = user_overlay

    if improved_keypoints:
        if improved_opacity < 1.0:
            cv2.addWeighted(improved_overlay, improved_opacity, frame, 1 - improved_opacity, 0, frame)
        else:
            frame[:] = improved_overlay


def render_feedback_panel(frame, worst_joints_feedback: List[str]) -> np.ndarray:
    panel_width = 600
    padding = 20
    font = config.FONT
    max_font_scale = 1
    min_font_scale = 0.75
    font_thickness = 1
    line_spacing = 40
    max_line_width = panel_width - 2 * padding

    h, w, _ = frame.shape
    font_scale = max_font_scale
    wrapped_lines = []
    total_text_height = h + 1  # force loop entry

    while font_scale >= min_font_scale:
        wrapped_lines.clear()
        for idx, feedback in enumerate(worst_joints_feedback, 1):
            numbered_text = f"{idx}. {feedback}"
            lines = wrap_text(numbered_text, max_line_width, font, font_scale, font_thickness)
            wrapped_lines.append(lines)

        total_lines = sum(len(lines) for lines in wrapped_lines)
        total_text_height = total_lines * line_spacing + (len(worst_joints_feedback) - 1) * int(line_spacing / 2)

        if total_text_height < h - 100:  # space for legend
            break

        font_scale -= 0.05

    # Determine desired height
    desired_height = max(total_text_height + 100, frame.shape[0])
    y_offset = (desired_height - frame.shape[0]) // 2

    # --- Create final composite canvas ---
    final_frame = np.ones((desired_height, w + panel_width, 3), dtype=np.uint8) * 255  # white bg
    final_frame[y_offset:y_offset + frame.shape[0], :w] = frame

    # --- Draw text panel separately and paste ---
    text_panel = np.ones((desired_height, panel_width, 3), dtype=np.uint8) * 255

    y_start = max((desired_height - total_text_height) // 2, padding)
    y = y_start
    for lines in wrapped_lines:
        for line in lines:
            cv2.putText(text_panel, line, (padding, y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            y += line_spacing
        y += int(line_spacing / 2)

    draw_legend(text_panel, 0, desired_height - 3 * 35 + 50)
    cv2.rectangle(text_panel, (0, 0), (text_panel.shape[1], text_panel.shape[0]), (0, 0, 0), config.BORDER_THICKNESS)
    final_frame[:, w:] = text_panel

    return final_frame



def render_improved_form_feedback(frame: np.ndarray,
                                  improved_keypoints: Dict[str, Tuple[int, int]],
                                  deviations,
                                  opacity_lines: float = 1.0,
                                  opacity_points: float = 1.0) -> np.ndarray:
    """
    Draws the improved skeleton in the center third of the frame with adjustable opacity.
    """
    h, w, _ = frame.shape

    overlay_lines = frame.copy()
    overlay_points = frame.copy()

    # Draw skeleton lines on line overlay
    for j1, j2 in JOINT_CONNECTIONS:
        if j1 in improved_keypoints and j2 in improved_keypoints:
            pt1 = improved_keypoints[j1]
            pt2 = improved_keypoints[j2]
            p1 = (pt1[0], pt1[1])
            p2 = (pt2[0], pt2[1])
            cv2.line(overlay_lines, p1, p2, (0, 0, 255), 2)  # red lines for improved form

    # Draw joint markers on point overlay
    for aspect in deviations.keys():
        for j in config.ANGLE_TO_JOINT.get(aspect, []):
            joint = j.upper()
            if joint in improved_keypoints:
                x, y = improved_keypoints[joint]
                new_x = x
                cv2.circle(overlay_points, (new_x, y), 6, (0, 255, 0), -1)  # green joints

    # Blend overlays with adjustable opacity
    blended = frame.copy()
    cv2.addWeighted(overlay_lines, opacity_lines, blended, 1 - opacity_lines, 0, blended)
    cv2.addWeighted(overlay_points, opacity_points, blended, 1 - opacity_points, 0, blended)

    draw_legend(blended, 0, h - config.LEGEND_HEIGHT, True)

    return blended

def wrap_text(text, max_width, font, font_scale, thickness):
    """
    Wraps a long string into lines so they fit within the given width.
    """
    words = text.split()
    lines = []
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        (text_width, _) = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if text_width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word

    if line:
        lines.append(line)

    return lines


import cv2
import numpy as np
import math
from src import config

def draw_filled_angle(
    img: np.ndarray,
    pivot: tuple,
    length: int = 20,
    angle_deg: float = 45.0,
    fill_color: tuple = (0, 255, 255),
    line_color: tuple = (0, 0, 0),
    thickness: int = 2
):
    """
    Draw a two-ray angle symbol with the wedge between them filled.
    - pivot: (x,y)
    - length: ray length
    - angle_deg: angle between rays
    - fill_color: BGR fill of wedge
    - line_color: BGR color for ray outlines
    """
    cx, cy = pivot
    # compute end of ray 1 (horizontal right)
    x1, y1 = cx + length, cy
    # compute end of ray 2 (rotated up by angle_deg)
    theta = math.radians(angle_deg)
    x2 = int(cx + length * math.cos(theta))
    y2 = int(cy - length * math.sin(theta))

    # fill the wedge triangle [pivot, ray1, ray2]
    pts = np.array([[cx, cy], [x1, y1], [x2, y2]], dtype=np.int32)
    cv2.fillPoly(img, [pts], fill_color)

    # draw the two rays in black
    cv2.line(img, (cx, cy), (x1, y1), line_color, thickness)
    cv2.line(img, (cx, cy), (x2, y2), line_color, thickness)


def draw_legend(new_frame: np.ndarray, x_start: int, y_start: int, ideal_frame: bool = False):
    """
    Draws a legend explaining color codes on the right panel.
    """
    font = config.FONT
    font_scale = 0.6
    thickness = 1
    spacing = 20

    h, w, _ = new_frame.shape
    cv2.rectangle(
        new_frame,
        (x_start, y_start),
        (w, y_start + config.LEGEND_HEIGHT),
        config.LEGEND_BACKGROUND_COLOR,
        -1
    )

    if ideal_frame:
        angle_pivot = 5
        # --- Row 1: yellow filled angle symbol = Ideal Angle ---
        cy1 = y_start + spacing - 3
        cx1 = x_start + 15
        draw_filled_angle(
            new_frame,
            pivot=(cx1 + 20, cy1 + angle_pivot),
            fill_color=(0, 255, 255),  # yellow fill
            line_color=(0, 0, 0),      # black outlines
            thickness=2
        )
        cv2.putText(
            new_frame,
            "= Ideal Angle",
            (x_start + 95, cy1 + 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

        # --- Row 2: red & green filled angle symbols = Original vs Improved Angles ---
        cy2 = y_start + 2 * spacing
        cx2 = x_start + 15
        # red (original)
        draw_filled_angle(
            new_frame,
            pivot=(cx2, cy2 + angle_pivot),
            fill_color=(0, 0, 255),    # red fill
            line_color=(0, 0, 0),
            thickness=2
        )
        # green (improved)
        draw_filled_angle(
            new_frame,
            pivot=(cx2 + 40, cy2 + angle_pivot),
            fill_color=(0, 255, 0),    # green fill
            line_color=(0, 0, 0),
            thickness=2
        )
        cv2.putText(
            new_frame,
            "= Your Angles",
            (cx2 + 80, cy2 + 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    else:
        items = [
            (config.FOCUS_JOINT_COLOR,
             f"Focused Points - Specifically for {config.CURRENT_TECHNIQUE.replace('_', ' ').title()}"),
            (config.WORST_JOINT_COLOR, "Worst Points - Focus on For Improvement"),
        ]
        for i, (color, label) in enumerate(items):
            radius = 8
            cy = y_start + (i + 1) * spacing
            cx = x_start + 15
            cv2.circle(new_frame, (cx, cy), radius, color, -1)
            cv2.putText(
                new_frame,
                label,
                (cx + 25, cy + 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        cv2.circle(
            new_frame,
            (x_start + 15, y_start + 2 * spacing),
            3,
            config.FOCUS_JOINT_COLOR,
            -1
        )
    cv2.rectangle(new_frame, (x_start, y_start), (w, h), (0, 0, 0), config.BORDER_THICKNESS)


def draw_feedback_connectors(
        frame,
        worst_joints: Dict[str, float],
        keypoints: Dict[str, Tuple[int, int]],
        start_index: int = 1
):
    """
    Draws feedback label circles around groups of joints, with lines connecting the label to each joint.
    """
    font = config.FONT
    font_scale = 0.6
    font_thickness = 2
    circle_radius_label = 14
    label_fill_color = (255, 255, 255)
    label_text_color = (0, 0, 0)
    outline_color = (0, 0, 0)
    line_color = (0, 0, 0)

    # Flatten all keypoints into one array for distance computation
    all_joint_coords = list(keypoints.values())

    for idx, aspect in enumerate(worst_joints.keys(), start=start_index):
        joints = config.ANGLE_TO_JOINT.get(aspect, [])

        joint_coords = [keypoints.get(j.upper()) for j in joints if j.upper() in keypoints]

        if not joint_coords:
            continue

        # 1. Get group center
        avg_x = int(sum(p[0] for p in joint_coords) / len(joint_coords))
        avg_y = int(sum(p[1] for p in joint_coords) / len(joint_coords))
        center = (avg_x, avg_y)

        # 2. Compute max distance to determine grouping radius
        radius = max(int(max(math.hypot(p[0] - avg_x, p[1] - avg_y) for p in joint_coords)) + 20, 100)

        # 3. Sample points around the circle, find one farthest from other keypoints
        best_point = None
        best_dist = -1
        for angle_deg in range(0, 360, 5):  # try points every 5 degrees
            angle_rad = math.radians(angle_deg)
            px = int(avg_x + radius * math.cos(angle_rad))
            py = int(avg_y + radius * math.sin(angle_rad))

            # Get min distance to all other keypoints (avoid overlapping them)
            min_dist = min(math.hypot(px - x, py - y) for (x, y) in all_joint_coords)

            if min_dist > best_dist:
                best_dist = min_dist
                best_point = (px, py)

        if not best_point:
            continue

        # 4. Draw connecting lines to each joint
        for j in joint_coords:
            cv2.line(frame, best_point, j, line_color, 2)

        # 5. Draw label circle at best_point
        cv2.circle(frame, best_point, circle_radius_label + 2, outline_color, thickness=3)
        cv2.circle(frame, best_point, circle_radius_label, label_fill_color, thickness=-1)
        cv2.putText(
            frame,
            str(idx),
            (best_point[0] - 6, best_point[1] + 6),
            font,
            font_scale,
            label_text_color,
            font_thickness,
            cv2.LINE_AA
        )

        all_joint_coords.append(best_point)


def add_header_to_frame(frame: np.ndarray, header_text: str,
                        height: int = 60,
                        bg_color: Tuple[int, int, int] = (240, 240, 240),
                        text_color: Tuple[int, int, int] = (0, 0, 0),
                        border_color: Tuple[int, int, int] = (0, 0, 0),
                        font=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale: float = 1.0,
                        thickness: int = 2) -> np.ndarray:
    """
    Adds a bordered header with centered text at the top of the frame.

    Args:
        frame (np.ndarray): Input image.
        header_text (str): Text to display in header.
        height (int): Height of the header area in pixels.
        bg_color (tuple): Background color of the header (BGR).
        text_color (tuple): Text color (BGR).
        border_color (tuple): Color of the header border (BGR).
        border_thickness (int): Thickness of the border lines.
        font: OpenCV font.
        font_scale (float): Font size.
        thickness (int): Text thickness.

    Returns:
        np.ndarray: Frame with header added on top.
    """
    h, w, _ = frame.shape

    # Create header bar
    header = np.ones((height, w, 3), dtype=np.uint8) * 255
    header[:] = bg_color

    # Add border to header
    cv2.rectangle(header, (0, 0), (w - 1, height - 1), border_color, config.BORDER_THICKNESS)

    # Calculate text size and position
    (text_width, text_height), _ = cv2.getTextSize(header_text, font, font_scale, thickness)
    x = (w - text_width) // 2
    y = (height + text_height) // 2 - 5

    # Draw the text
    cv2.putText(header, header_text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Stack header and frame
    framed_with_header = np.vstack([header, frame])

    return framed_with_header


