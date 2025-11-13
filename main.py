from fastapi import FastAPI
from fastapi.responses import JSONResponse # For clearer success/error messages
import os
import sys
from pathlib import Path
from threading import Thread
import time

app = FastAPI()

# --- Configuration for relative imports ---
# This inserts the directory containing the current script (app.py) 
# followed by the 'scripts' subdirectory into the Python path.
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
scripts_dir = os.path.join(base_dir, 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
# -----------------------------------------

# Import the core modules after path setup
try:
    # Assuming 'scripts' contains these files
    from video_to_frames import process_all_videos
    from detect_person import process_all_frame_folders as detect_persons_in_folders
    from silhouette_extraction import SilhouetteExtractor
except ImportError as e:
    # If modules are not found, print a clear error message
    print(f"CRITICAL ERROR: Could not import required scripts: {e}")
    print("Ensure 'video_to_frames.py', 'detect_person.py', and 'silhouette_extraction.py' are in a 'scripts' subdirectory.")
    
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

def setup_directories():
    """Create necessary directories"""
    # ... (setup_directories function remains the same) ...
    directories = [
        'data/videos',
        'data/frames',
        'data/detected_persons',
        'data/silhouettes',
        'output'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories created/verified")


def main():
    
    # Define paths
    video_dir = 'data/videos'
    frames_dir = 'data/frames'
    detected_persons_dir = 'data/detected_persons'
    silhouettes_dir = 'data/silhouettes'

    setup_directories()
    
    # Check if videos exist
    video_dir_path = Path(video_dir)
    # Check for person folders (nested structure)
    person_folders = [f for f in video_dir_path.iterdir() if f.is_dir()]
    
    # Check for videos in person folders or directly
    video_extensions = ['.mp4', '.avi', '.mov', '.flv', '.mkv']
    has_videos = False
    
    if person_folders:
        for person_folder in person_folders:
            videos_in_folder = [f for f in person_folder.iterdir() 
                              if f.suffix.lower() in video_extensions]
            if videos_in_folder:
                has_videos = True
                break
    else:
        direct_videos = [f for f in video_dir_path.glob('*.*') 
                         if f.suffix.lower() in video_extensions]
        has_videos = len(direct_videos) > 0
    
    if has_videos:
        frame_interval = 1 
        process_all_videos(video_dir, frames_dir, frame_interval)
    else:
        print(f"⚠ No video files found in {video_dir}. Pipeline stopped.")
        # Removed return here to allow continued execution if frames already exist
    
    # Check if frames exist
    frames = list(Path(frames_dir).glob('**/*.*'))
    if frames:
        detection_model = 'yolo11n-pose.pt' 
        detection_conf = 0.3 
        device = 'cpu' 
        save_bbox_images = True 
        save_crops = True 
        
        detect_persons_in_folders(
            frames_dir, 
            detected_persons_dir, 
            model=detection_model, 
            conf=detection_conf, 
            device=device,
            save_bbox_images=save_bbox_images,
            save_crops=save_crops
        )
    else:
        print(f"⚠ No frame files found in {frames_dir}. Skipping detection.")
    
    # Check if detected persons exist
    detected_crops_dir = Path(detected_persons_dir)
    crop_folders = list(detected_crops_dir.glob('**/person_crops'))
    
    if crop_folders:
        # Configuration
        method = 'yolo'
        silhouette_model = 'yolo11n-seg.pt' 
        silhouette_conf = 0.25
        
        # Process each person's crops folder
        for crop_folder in crop_folders:
            relative_path = crop_folder.relative_to(detected_crops_dir)
            output_path = Path(silhouettes_dir) / relative_path.parent
            
            print(f"\nProcessing: {relative_path.parent}")
            
            # Initialize Extractor for this folder (or initialize once outside the loop 
            # if the extractor class can handle multiple runs efficiently)
            if method == 'yolo':
                extractor = SilhouetteExtractor(method=method, model=silhouette_model, conf=silhouette_conf, device=device)
            else:
                extractor = SilhouetteExtractor(method=method)
            
            extractor.extract_silhouettes_from_folder(str(crop_folder), str(output_path))
    else:
        print(f"⚠ No detected person crops found in {detected_persons_dir}. Skipping silhouette extraction.")
        
    # ... (summary print statements remain the same) ...


if __name__ == "__main__":
    print("Server initialized. Run using 'uvicorn your_file_name:app --reload'")