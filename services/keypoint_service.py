import os
from pathlib import Path
from models.keypoints_model import PersonData, VideoData
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

        videos = []
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
            
            frames = []
            for csv_file in csv_files:
                try:
                    # parse_keypoints_csv returns a list of FrameData
                    frame_data_list = parse_keypoints_csv(str(csv_file))
                    frames.extend(frame_data_list)
                except Exception as e:
                    print(f"  ⚠ Error parsing {csv_file.name}: {e}")
                    continue

            if frames:
                videos.append(VideoData(video_name=video_name, frames=frames))
                print(f"  ✓ Loaded {len(frames)} frames from {video_name}")

        if not videos:
            print(f"⚠ No video keypoint data found for {person_name}")
            return 0

        person_data = PersonData(person_name=person_name, videos=videos)
        self.collection.insert_one(person_data.model_dump())
        print(f"✓ Saved data for {person_name} ({len(videos)} videos)")
        return len(videos)
