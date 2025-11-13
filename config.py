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
    
    # Folder Structure
    VIDEOS_DIR = os.getenv('VIDEOS_DIR', 'videos')
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    
    # Processing Configuration
    FRAME_INTERVAL = int(os.getenv('FRAME_INTERVAL', '1'))
    SAVE_BBOX_IMAGES = os.getenv('SAVE_BBOX_IMAGES', 'true').lower() == 'true'
    SAVE_CROPS = os.getenv('SAVE_CROPS', 'true').lower() == 'true'
    SILHOUETTE_METHOD = os.getenv('SILHOUETTE_METHOD', 'yolo')
    
    @classmethod
    def get_person_folders(cls, person_name):

        person_data_dir = os.path.join(cls.DATA_DIR, person_name)
        
        return {
            'videos': os.path.join(cls.VIDEOS_DIR, person_name),
            'frames': os.path.join(person_data_dir, 'frames'),
            'detected_persons': os.path.join(person_data_dir, 'detected_persons'),
            'silhouettes': os.path.join(person_data_dir, 'silhouettes')
        }



# Create config instance
config = Config()
