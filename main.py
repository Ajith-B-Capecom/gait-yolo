from fastapi import FastAPI
from fastapi.responses import JSONResponse # For clearer success/error messages
import os
import sys
from pathlib import Path
from threading import Thread
import time
from services.keypoint_service import KeypointsService
from scripts.video_to_frames import process_all_videos
from scripts.detect_person import process_all_frame_folders as detect_persons_in_folders
from scripts.silhouette_extraction import SilhouetteExtractor, process_all_frame_folders as extract_silhouettes_from_folders
from config import config
app = FastAPI()
service = KeypointsService()

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
scripts_dir = os.path.join(base_dir, 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# --- FastAPI Endpoints ---

@app.get("/")
def read_root():
    return {"Hello": "FastAPI is running! Use /start_pipeline to process data."}

@app.get("/start_pipeline")
async def start_pipeline():
    try:
        # Start the heavy processing in a non-blocking thread
        thread = Thread(target=main)
        thread.start()
        
        return JSONResponse(
            status_code=202, # 202 Accepted status for async processing
            content={"message": "✅ Pipeline started successfully in the background. Check server logs for progress."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"❌ Failed to start pipeline: {e}"}
        )

# --- Core Processing Logic ---

def setup_directories_for_person(person_name):
    """Create necessary directories for a person"""
    # New structure: videos/person1/ and data/person1/frames/, etc.
    person_video_dir = Path(config.VIDEOS_DIR) / person_name
    person_data_dir = Path(config.DATA_DIR) / person_name
    
    directories = [
        person_video_dir,
        person_data_dir / 'frames',
        person_data_dir / 'detected_persons',
        person_data_dir / 'silhouettes'
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Directories created/verified for {person_name}")


def main():

    # Get all persons from videos folder
    video_dir_path = Path(config.VIDEOS_DIR)
    
    if not video_dir_path.exists():
        print(f"\n✗ Videos directory not found: {config.VIDEOS_DIR}/")
        print(f"  Creating directory...")
        video_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"\n  Please add videos to: {config.VIDEOS_DIR}/person1/, person2/, etc.")
        return
    
    # Find all person folders
    person_folders = [f for f in video_dir_path.iterdir() if f.is_dir() and f.name.startswith('person')]
    
    if not person_folders:
        print(f"\n✗ No person folders found in {config.VIDEOS_DIR}/")
        print(f"\nExpected structure:")
        print(f"  {config.VIDEOS_DIR}/person1/video.mp4")
        print(f"  {config.VIDEOS_DIR}/person2/video.mp4")
        return
    
    print(f"\nFound {len(person_folders)} person(s): {[f.name for f in person_folders]}")
    
    # Process each person
    for person_folder in person_folders:
        person_name = person_folder.name
        
        print("\n" + "="*70)
        print(f"  Processing: {person_name.upper()}")
        print("="*70)
        
        # Setup directories for this person
        setup_directories_for_person(person_name)
        
        # Define paths for this person
        person_video_dir = Path(config.VIDEOS_DIR) / person_name
        person_data_dir = Path(config.DATA_DIR) / person_name
        person_frames_dir = person_data_dir / 'frames'
        person_detected_dir = person_data_dir / 'detected_persons'
        person_silhouettes_dir = person_data_dir / 'silhouettes'
        
        # Check for videos
        video_extensions = ['.mp4', '.avi', '.mov', '.flv', '.mkv']
        videos = [f for f in person_video_dir.glob('*.*') if f.suffix.lower() in video_extensions]
        
        if not videos:
            print(f"⚠ No videos found for {person_name}")
            continue
        
        print(f"\nFound {len(videos)} video(s) for {person_name}")
        
        # STEP 1: Extract frames
        print("\n" + "-"*70)
        print("STEP 1: Video to Frames Extraction")
        print("-"*70)
        
        process_all_videos(
            str(person_video_dir),
            str(person_frames_dir),
            config.FRAME_INTERVAL
        )
        
        # STEP 2: Detect persons
        print("\n" + "-"*70)
        print("STEP 2: Person Detection + Pose Estimation")
        print("-"*70)
        
        frames = list(person_frames_dir.glob('**/*.jpg'))
        if frames:
            detect_persons_in_folders(
                str(person_frames_dir),
                str(person_detected_dir),
                model=config.YOLO_DETECTION_MODEL,
                conf=config.DETECTION_CONFIDENCE,
                device=config.DEVICE,
                save_bbox_images=config.SAVE_BBOX_IMAGES,
                save_crops=config.SAVE_CROPS
            )
        else:
            print(f"⚠ No frames found for {person_name}")
            continue
        
        try:
            total_saved = service.save_person_videos(person_name, str(person_detected_dir))
            if total_saved > 0:
                print(f"✓ Saved {total_saved} video(s) keypoint records to MongoDB for {person_name}")
            else:
                print(f"⚠ No keypoint data found for {person_name}")
        except Exception as e:
            print(f"⚠ MongoDB save failed: {e}")
        
        # STEP 3: Extract silhouettes
        print("\n" + "-"*70)
        print("STEP 3: Silhouette Extraction")
        print("-"*70)
        
        crop_folders = list(person_detected_dir.glob('**/person_crops'))
        
        if crop_folders:
            for crop_folder in crop_folders:
                relative_path = crop_folder.relative_to(person_detected_dir)
                output_path = person_silhouettes_dir / relative_path.parent
                
                print(f"\nProcessing: {relative_path.parent}")
                
                if config.SILHOUETTE_METHOD == 'yolo':
                    extractor = SilhouetteExtractor(
                        method='yolo',
                        model=config.YOLO_SEGMENTATION_MODEL,
                        conf=config.SEGMENTATION_CONFIDENCE,
                        device=config.DEVICE
                    )
                else:
                    extractor = SilhouetteExtractor(method=config.SILHOUETTE_METHOD)
                
                extractor.extract_silhouettes_from_folder(str(crop_folder), str(output_path))
        else:
            print(f"⚠ No detected person crops found for {person_name}")

if __name__ == "__main__":
    print("Server initialized. Run using 'uvicorn your_file_name:app --reload'")