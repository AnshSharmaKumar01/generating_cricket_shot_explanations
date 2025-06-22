import cv2
import mediapipe as mp

from src.utils.helpers import get_pose_bounding_box

# Initialize MediaPipe pose estimator
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints_from_frame(frame):
    """
    Extract pose keypoints from a single frame using MediaPipe.
    Returns a dictionary of joint_name -> (x, y) coordinates (normalized to frame size).
    """
    height, width, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    keypoints = {}
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            x, y = int(lm.x * width), int(lm.y * height)
            keypoints[mp_pose.PoseLandmark(idx).name] = (x, y)
    return keypoints


def draw_keypoints_on_frame(frame, keypoints):
    """
    Annotates the frame with keypoints.
    """
    for joint_name, (x, y) in keypoints.items():
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, joint_name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame


def get_frame_from_video(video_path, target_frame=120):
    """
    Load a specific frame from the video.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"Failed to read frame {target_frame}")
    return frame


def extract_keypoints(video_path, target_frame):
    frame = get_frame_from_video(video_path, target_frame=target_frame)
    tmp_keypoints = extract_keypoints_from_frame(frame)

    x_min, y_min, x_max, y_max = get_pose_bounding_box(tmp_keypoints)
    cropped_player = frame[y_min:y_max, x_min:x_max]

    # cv2.imshow("TEST", cropped_player)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    keypoints = extract_keypoints_from_frame(cropped_player)

    return cropped_player, keypoints


if __name__ == "__main__":
    from src import config

    video_path = "sample_videos/ideal_videos/cover_drive.mp4"
    target_frame = 837

    frame, keypoints = extract_keypoints(video_path, target_frame)
    annotated_frame = draw_keypoints_on_frame(frame, keypoints)

    print("Extracted keypoints:")
    for name, coord in keypoints.items():
        print(f"{name}: {coord}")

    cv2.imshow("Pose Estimation - Frame 120", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
