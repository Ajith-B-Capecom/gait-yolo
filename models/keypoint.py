"""
Keypoint Model
Database model for storing keypoint data
"""

from datetime import datetime
from typing import Dict, List, Optional


class Keypoint:
    """Keypoint data model"""
    
    # COCO 17 keypoint names
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(
        self,
        person_name: str,
        frame: str,
        keypoint_id: int,
        keypoint_name: str,
        x: float,
        y: float,
        confidence: float,
        visible: str,
        video_name: Optional[str] = None,
        csv_source: Optional[str] = None
    ):
        """
        Initialize Keypoint model
        
        Args:
            person_name: Person identifier (e.g., 'person1')
            frame: Frame identifier (e.g., 'frame_000000')
            keypoint_id: Keypoint ID (0-16)
            keypoint_name: Keypoint name (e.g., 'nose')
            x: X coordinate
            y: Y coordinate
            confidence: Detection confidence (0-1)
            visible: Visibility ('yes' or 'no')
            video_name: Video identifier (optional)
            csv_source: Source CSV file path (optional)
        """
        self.person_name = person_name
        self.frame = frame
        self.keypoint_id = keypoint_id
        self.keypoint_name = keypoint_name
        self.x = float(x)
        self.y = float(y)
        self.confidence = float(confidence)
        self.visible = visible
        self.video_name = video_name
        self.csv_source = csv_source
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB"""
        return {
            'person_name': self.person_name,
            'frame': self.frame,
            'keypoint_id': self.keypoint_id,
            'keypoint_name': self.keypoint_name,
            'x': self.x,
            'y': self.y,
            'confidence': self.confidence,
            'visible': self.visible,
            'video_name': self.video_name,
            'csv_source': self.csv_source,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Keypoint':
        """Create Keypoint from dictionary"""
        return cls(
            person_name=data['person_name'],
            frame=data['frame'],
            keypoint_id=data['keypoint_id'],
            keypoint_name=data['keypoint_name'],
            x=data['x'],
            y=data['y'],
            confidence=data['confidence'],
            visible=data['visible'],
            video_name=data.get('video_name'),
            csv_source=data.get('csv_source')
        )
    
    @classmethod
    def from_csv_row(cls, row: Dict, person_name: str, video_name: Optional[str] = None, csv_source: Optional[str] = None) -> 'Keypoint':
        """
        Create Keypoint from CSV row
        
        Args:
            row: CSV row as dictionary
            person_name: Person identifier
            video_name: Video identifier (optional)
            csv_source: Source CSV file path (optional)
        
        Returns:
            Keypoint instance
        """
        return cls(
            person_name=person_name,
            frame=row['frame'],
            keypoint_id=int(row['keypoint_id']),
            keypoint_name=row['keypoint_name'],
            x=float(row['x']),
            y=float(row['y']),
            confidence=float(row['confidence']),
            visible=row['visible'],
            video_name=video_name,
            csv_source=csv_source
        )
    
    def __repr__(self) -> str:
        return f"Keypoint({self.person_name}, {self.frame}, {self.keypoint_name}, x={self.x:.2f}, y={self.y:.2f})"
