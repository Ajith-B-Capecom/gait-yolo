import os
from pathlib import Path
from models.keypoints_model import FrameData
from utils.csv_parser import parse_keypoints_csv
from utils.db import get_db


class KeypointsService:
    def __init__(self):
        self.db = get_db()
        self.collection = self.db["keypoints"]

    def save_person_videos(self, person_name: str, base_path: str):
        """
        Process all videos for a given person, convert CSV keypoints → MongoDB
        base_path should be: data/person1/detected_persons/
        """
        person_folder = Path(base_path)
        if not person_folder.exists():
            print(f"⚠ Person folder not found: {person_folder}")
            return 0

        total_frames = 0
        # Look for video folders inside detected_persons
        for video_dir in person_folder.glob("*"):
            if not video_dir.is_dir():
                continue

            video_name = video_dir.name
            
            # Look for CSV files in person_crops subfolder
            person_crops_dir = video_dir / "person_crops"
            if not person_crops_dir.exists():
                continue
                
            csv_files = list(person_crops_dir.glob("*_keypoints.csv"))
            
            if not csv_files:
                continue
            
            for csv_file in csv_files:
                try:
                    # parse_keypoints_csv returns a list of FrameData
                    frame_data_list = parse_keypoints_csv(
                        str(csv_file), 
                        video_name=video_name, 
                        person_name=person_name
                    )
                    
                    # Insert each frame into MongoDB
                    for frame_data in frame_data_list:
                        self.collection.insert_one(frame_data.model_dump())
                    
                    total_frames += len(frame_data_list)
                    print(f"  ✓ Loaded {len(frame_data_list)} frames from {video_name}/{csv_file.name}")
                except Exception as e:
                    print(f"  ⚠ Error parsing {csv_file.name}: {e}")
                    continue

        if total_frames == 0:
            print(f"⚠ No video keypoint data found for {person_name}")
            return 0

        print(f"✓ Saved {total_frames} frames for {person_name}")
        return total_frames
