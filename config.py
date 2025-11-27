"""
Configuration Management
Loads settings from .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for OpenGait pipeline"""
    
    # YOLO Model Configuration
    YOLO_DETECTION_MODEL = os.getenv('YOLO_DETECTION_MODEL', 'yolo11n-pose.pt')
    YOLO_SEGMENTATION_MODEL = os.getenv('YOLO_SEGMENTATION_MODEL', 'yolo11n-seg.pt')
    DETECTION_CONFIDENCE = float(os.getenv('DETECTION_CONFIDENCE', '0.3'))
    SEGMENTATION_CONFIDENCE = float(os.getenv('SEGMENTATION_CONFIDENCE', '0.25'))
    DEVICE = os.getenv('DEVICE', 'cpu')
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'gait')
    MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'keypoints')


# Create config instance
config = Config()
