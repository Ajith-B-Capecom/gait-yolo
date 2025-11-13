import pandas as pd
from models.keypoints_model import Keypoint, BodyPart, FrameData
from typing import List, Optional

def parse_keypoints_csv(csv_path: str, video_name: str = "", person_name: str = "") -> List[FrameData]:
    """
    Parse a person's keypoints CSV file.
    CSV format: Each row = one frame with keypoints for ONE person
    Returns: List of FrameData (one per frame)
    """
    df = pd.read_csv(csv_path)
    frames = []

    for _, row in df.iterrows():
        frame_name = row["frame"] if "frame" in df.columns else csv_path.split("\\")[-1].replace(".csv", "")
        
        # Create keypoints for this person in this frame
        kp = BodyPart(
            nose=Keypoint(x=row["x_nose"], y=row["y_nose"], conf=row["conf_nose"]),
            left_eye=Keypoint(x=row["x_left_eye"], y=row["y_left_eye"], conf=row["conf_left_eye"]),
            right_eye=Keypoint(x=row["x_right_eye"], y=row["y_right_eye"], conf=row["conf_right_eye"]),
            left_ear=Keypoint(x=row["x_left_ear"], y=row["y_left_ear"], conf=row["conf_left_ear"]),
            right_ear=Keypoint(x=row["x_right_ear"], y=row["y_right_ear"], conf=row["conf_right_ear"]),
            left_shoulder=Keypoint(x=row["x_left_shoulder"], y=row["y_left_shoulder"], conf=row["conf_left_shoulder"]),
            right_shoulder=Keypoint(x=row["x_right_shoulder"], y=row["y_right_shoulder"], conf=row["conf_right_shoulder"]),
            left_elbow=Keypoint(x=row["x_left_elbow"], y=row["y_left_elbow"], conf=row["conf_left_elbow"]),
            right_elbow=Keypoint(x=row["x_right_elbow"], y=row["y_right_elbow"], conf=row["conf_right_elbow"]),
            left_wrist=Keypoint(x=row["x_left_wrist"], y=row["y_left_wrist"], conf=row["conf_left_wrist"]),
            right_wrist=Keypoint(x=row["x_right_wrist"], y=row["y_right_wrist"], conf=row["conf_right_wrist"]),
            left_hip=Keypoint(x=row["x_left_hip"], y=row["y_left_hip"], conf=row["conf_left_hip"]),
            right_hip=Keypoint(x=row["x_right_hip"], y=row["y_right_hip"], conf=row["conf_right_hip"]),
            left_knee=Keypoint(x=row["x_left_knee"], y=row["y_left_knee"], conf=row["conf_left_knee"]),
            right_knee=Keypoint(x=row["x_right_knee"], y=row["y_right_knee"], conf=row["conf_right_knee"]),
            left_ankle=Keypoint(x=row["x_left_ankle"], y=row["y_left_ankle"], conf=row["conf_left_ankle"]),
            right_ankle=Keypoint(x=row["x_right_ankle"], y=row["y_right_ankle"], conf=row["conf_right_ankle"]),
        )
        
        # Each row is a different frame with one person
        frames.append(FrameData(
            frame=frame_name, 
            persons=[kp],
            video_name=video_name,
            person_name=person_name
        ))

    return frames
