import os

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FOCUS_POINTS = [10]
FOCUS_POINTS = [1, 3, 5, 10]  #TODO - Uncomment this when producing prototypes

TECHNIQUES_MAP = {
    "cover_drive": "Cover Drive",
    "bowled": "Bowled",
    "defence": "Defence",
    "pull_shot": "Pull Shot",
    "reverse_sweep": "Reverse Sweep"
}
EXPERTISES = ["Beginner", "Intermediate", "Expert"]

CURRENT_TECHNIQUE = "cover_drive"
CURRENT_EXPERTISE = "Beginner"

TARGET_FRAME_MAP = {
    ("ideal_videos", "cover_drive"): 837,
    ("practice_videos", "cover_drive"): 69,
}

LOW_THRESHOLD = 15
MEDIUM_THRESHOLD = 30

ANGLE_TO_JOINT = {
    "front_knee_angle": ["left_hip", "left_knee", "left_ankle"],
    "back_knee_angle": ["right_hip", "right_knee", "right_ankle"],
    "front_elbow_angle": ["left_shoulder", "left_elbow", "left_wrist"],
    "back_elbow_angle": ["right_shoulder", "right_elbow", "right_wrist"],
    "shoulder_rotation": ["left_shoulder", "right_shoulder"],
    "hip_rotation": ["left_hip", "right_hip"],
    "spine_forward_lean": ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
    "head_tilt": ["left_ear", "right_ear"],
    "bat_angle": ["right_wrist", "right_thumb"]
}

ANGLE_COMPONENTS = {
    "front_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "back_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "front_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "back_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "shoulder_rotation": ("left_shoulder", "neck", "right_shoulder"),
    "hip_rotation": ("left_hip", "spine", "right_hip"),
    "spine_forward_lean": ("left_hip", "spine", "left_shoulder"),
    "head_tilt": ("left_ear", "nose", "right_ear"),
    "bat_angle": ("right_elbow", "right_wrist", "bat_tip")
}

BASIC_JOINT_COLOR = (255, 0, 0)
FOCUS_JOINT_COLOR = (0, 255, 255)
WORST_JOINT_COLOR = (0, 0, 255)
LINE_COLOR = (100, 100, 100)

IDEAL_ANGLE_COLOR = (0, 255, 255)

FONT = cv2.FONT_HERSHEY_DUPLEX

BORDER_THICKNESS = 5

LEGEND_BACKGROUND_COLOR = (100, 100, 100)
LEGEND_HEIGHT = 55