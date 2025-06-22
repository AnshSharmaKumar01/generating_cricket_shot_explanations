import os

import cv2

from src import config
from src.explanation.explanation_controller import make_explanation

root = config.PROJECT_ROOT

techniques = config.TECHNIQUES_MAP.keys()
expertises = config.EXPERTISES
focus_points_list = config.FOCUS_POINTS

sample_video_types = ["ideal_videos", "practice_videos"]

not_implemented_techniques = ["bowled", "defence", "pull_shot", "reverse_sweep"]
not_implemented_video_type = ["practice_videos"]
skip_these = ["Beginner"]

if __name__ == '__main__':
    for technique in techniques:
        if technique in not_implemented_techniques:  #TODO - Implement different techniques and delete this
            continue
        for sample_video_type in sample_video_types:
            if sample_video_type in not_implemented_video_type:
                continue
            for expertise in expertises:
                if expertise in skip_these:
                    continue
                for focus_points in focus_points_list:
                    print(f"""
                    ########################################################################
                    RUNNING CONFIG - {technique} - {sample_video_type} - {expertise} - {focus_points}
                    ########################################################################
                    """)
                    config.CURRENT_TECHNIQUE = technique
                    config.CURRENT_EXPERTISE = expertise

                    display_frame = make_explanation(technique, sample_video_type, expertise, focus_points, root)

                    image_dir = f"prototypes/{sample_video_type}/{technique}/{expertise}"
                    image_file = f"{focus_points}_points_feedback.jpg"

                    image_path = os.path.join(image_dir, image_file)

                    os.makedirs(image_dir, exist_ok=True)

                    cv2.imwrite(image_path, display_frame)
                    cv2.imshow(f"{config.TECHNIQUES_MAP.get(technique)} {expertise} Feedback",
                               display_frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print(f"""
                    ########################################################################
                    SUCCESSFUL CONFIG - {technique} - {sample_video_type} - {expertise} - {focus_points}
                    ########################################################################
                    """)