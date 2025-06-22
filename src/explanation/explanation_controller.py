import os

from src import config
from src.evaluation.ideal_pose_profiles import get_ideal_pose
from src.evaluation.joint_metrics import calculate_joint_angles, compute_joint_deviation
from src.explanation.beginner_feedback import rank_worst_joints
from src.pose_estimation.mediapipe_pose import extract_keypoints
from src.explanation import beginner_feedback as beginner
from src.explanation import intermediate_feedback as intermediate
from src.explanation import expert_feedback as expert


def get_target_frame(s, t):
    return config.TARGET_FRAME_MAP.get((s, t))


def get_video_path(s, t, r):
    return os.path.join(r, "Research Project Implementation/sample_videos", s,
                        t + ".mp4")


def make_explanation(technique, sample_video_type, expertise, focus_points, root):
    target_frame = get_target_frame(sample_video_type, technique)
    video_path = get_video_path(sample_video_type, technique, root)

    # Step 1: Pose Extraction
    player_frame, keypoints = extract_keypoints(video_path, target_frame)

    # Step 2: Ideal Pose
    ideal_pose = get_ideal_pose(technique, root)

    # Step 3: Explanation for frame
    user_angles = calculate_joint_angles(keypoints)
    deviations = compute_joint_deviation(user_angles, ideal_pose)
    worst_joints = rank_worst_joints(deviations, keypoints, top_k=focus_points)

    if expertise == "Beginner":
        return beginner.generate_feedback(technique, player_frame, worst_joints, keypoints, deviations)
    if expertise == "Intermediate":
        return intermediate.generate_feedback(technique, player_frame, worst_joints, keypoints, deviations,
                                              user_angles, ideal_pose)
    if expertise == "Expert":
        return expert.generate_feedback(
            technique=technique,
            player_frame=player_frame,
            worst_joints=worst_joints,
            keypoints=keypoints,
            deviations=deviations,    # unused internally but kept for signature consistency
            user_angles=user_angles,
            ideal_pose=ideal_pose
        )
