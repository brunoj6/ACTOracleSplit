from typing import Dict, List

# Example joint index mapping inside the 14-D vector:
LEFT_IDXS:  List[int] = [0,1,2,3,4,5,6]    # adjust to your real layout
RIGHT_IDXS: List[int] = [7,8,9,10,11,12,13]

AGENT_SPECS: Dict[str, Dict] = {
    "left": {
        "joint_idxs": LEFT_IDXS,
        "camera_names": ["left_wrist", "cam_high", "cam_right_wrist"] # ["left_wrist", "fixed"]  # or keep all if desired
    },
    "right": {
        "joint_idxs": RIGHT_IDXS,
        "camera_names": ["left_wrist", "cam_high", "cam_right_wrist"] # ["right_wrist", "fixed"]
    },
}