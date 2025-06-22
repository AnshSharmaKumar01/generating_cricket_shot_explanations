import json
import os


def get_ideal_pose(technique: str, root) -> dict:
    IDEAL_POSE_FILE = os.path.join(root, "Research Project Implementation/src/assets", "ideal_poses.json")
    try:
        with open(IDEAL_POSE_FILE, "r") as f:
            all_poses = json.load(f)

        if technique not in all_poses:
            raise ValueError(f"No ideal pose data found for technique: {technique}")

        return all_poses[technique]

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find ideal pose file at: {IDEAL_POSE_FILE}")
    except json.JSONDecodeError:
        raise ValueError("ideal_poses.json is not properly formatted.")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(get_ideal_pose("cover_drive", root))
