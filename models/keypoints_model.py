from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Keypoint(BaseModel):
    x: float
    y: float
    conf: float


class BodyPart(BaseModel):
    nose: Optional[Keypoint] = None
    left_eye: Optional[Keypoint] = None
    right_eye: Optional[Keypoint] = None
    left_ear: Optional[Keypoint] = None
    right_ear: Optional[Keypoint] = None
    left_shoulder: Optional[Keypoint] = None
    right_shoulder: Optional[Keypoint] = None
    left_elbow: Optional[Keypoint] = None
    right_elbow: Optional[Keypoint] = None
    left_wrist: Optional[Keypoint] = None
    right_wrist: Optional[Keypoint] = None
    left_hip: Optional[Keypoint] = None
    right_hip: Optional[Keypoint] = None
    left_knee: Optional[Keypoint] = None
    right_knee: Optional[Keypoint] = None
    left_ankle: Optional[Keypoint] = None
    right_ankle: Optional[Keypoint] = None


class FrameData(BaseModel):
    frame: str
    persons: List[BodyPart]
    video_name: str
    person_name: str
    


# class VideoData(BaseModel):
#     video_name: str
#     frames: List[FrameData]


# class PersonData(BaseModel):
#     person_name: str
#     videos: List[VideoData]
