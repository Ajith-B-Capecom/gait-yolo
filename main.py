"""
OpenGAIT Processing Pipeline
Main script to orchestrate video to frames to silhouettes extraction
"""

import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from video_to_frames import extract_frames, process_all_videos
from detect_person import PersonDetector, process_all_frame_folders as detect_persons_in_folders
from silhouette_extraction import SilhouetteExtractor, process_all_frame_folders as extract_silhouettes_from_folders


def setup_directories():
    """Create necessary directories"""
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
    """Main processing pipeline"""
    
    print("\n" + "="*60)
    print("  OpenGAIT Processing Pipeline")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Define paths
    video_dir = 'data/videos'
    frames_dir = 'data/frames'
    detected_persons_dir = 'data/detected_persons'
    silhouettes_dir = 'data/silhouettes'
    
    print(f"\nVideo directory: {video_dir}")
    print(f"Frames output: {frames_dir}")
    print(f"Detected persons output: {detected_persons_dir}")
    print(f"Silhouettes output: {silhouettes_dir}")
    
    print("\n" + "="*60)
    print("STEP 1: Video to Frames Extraction")
    print("="*60)
    
    # Check if videos exist
    video_dir_path = Path(video_dir)
    
    # Check for person folders
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
        frame_interval = 1  # Change this to extract every nth frame
        process_all_videos(video_dir, frames_dir, frame_interval)
    else:
        print(f"⚠ No video files found in {video_dir}")
        print("\nExpected structure:")
        print("  data/videos/person1/video.mp4")
        print("  data/videos/person2/video.mp4")
        print("\nOr flat structure:")
        print("  data/videos/video1.mp4")
        print("  data/videos/video2.mp4")
        print("\nPlease add video files and run again.")
        return
    
    print("\n" + "="*60)
    print("STEP 2: Person Detection (YOLOv11)")
    print("="*60)
    
    # Check if frames exist
    frames = list(Path(frames_dir).glob('**/*.*'))
    if frames:
        # YOLO detection configuration
        detection_model = 'yolo11n-pose.pt'  # Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
        detection_conf = 0.5
        device = 'cpu'  # Use 'cuda' or '0' for GPU acceleration
        save_bbox_images = True  # Save visualization with bounding boxes
        save_crops = True  # Save cropped person images
        
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
        print(f"⚠ No frame files found in {frames_dir}")
        print("Please run frame extraction first")
        return
    
    print("\n" + "="*60)
    print("STEP 3: Silhouette Extraction")
    print("="*60)
    
    # Check if detected persons exist
    detected_crops_dir = Path(detected_persons_dir)
    crop_folders = list(detected_crops_dir.glob('**/person_crops'))
    
    if crop_folders:
        # Extract silhouettes from cropped person images
        # Use YOLO segmentation or background subtraction
        
        # Option 1: YOLO segmentation (best quality)
        method = 'yolo'
        silhouette_model = 'yolo11n-seg.pt'  # Options: yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg
        silhouette_conf = 0.25
        
        # Option 2: Background subtraction (faster but lower quality)
        # method = 'mog2'  # or 'knn'
        
        # Process each person's crops folder
        for crop_folder in crop_folders:
            # Get relative path structure
            relative_path = crop_folder.relative_to(detected_crops_dir)
            output_path = Path(silhouettes_dir) / relative_path.parent
            
            print(f"\nProcessing: {relative_path.parent}")
            
            if method == 'yolo':
                extractor = SilhouetteExtractor(method=method, model=silhouette_model, conf=silhouette_conf, device=device)
            else:
                extractor = SilhouetteExtractor(method=method)
            
            extractor.extract_silhouettes_from_folder(str(crop_folder), str(output_path))
    else:
        print(f"⚠ No detected person crops found in {detected_persons_dir}")
        print("Please run person detection first")
        return
    
    print("\n" + "="*60)
    print("✓ Processing Complete!")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - Frames: {frames_dir}")
    print(f"  - Detected Persons: {detected_persons_dir}")
    print(f"    • Bounding box visualizations: {detected_persons_dir}/**/bbox_visualizations/")
    print(f"    • Cropped persons: {detected_persons_dir}/**/person_crops/")
    print(f"  - Silhouettes: {silhouettes_dir}")


if __name__ == "__main__":
    main()
