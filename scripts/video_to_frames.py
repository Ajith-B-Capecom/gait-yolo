
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np


def extract_frames(video_path, output_dir, frame_interval=1):

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo Information:")
    print(f"  Total Frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    frame_count = 0
    extracted_count = 0
    
    print(f"\nExtracting frames from: {video_path}")
    
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract every nth frame
            if frame_count % frame_interval == 0:
                frame_name = f"frame_{extracted_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()    #Frees the video resource from memory after completion.
    
    print(f"\nFrames Extracted: {extracted_count}")
    print(f"Saved to: {output_dir}")
    
    return extracted_count


def process_all_videos(video_dir, output_base_dir, frame_interval=1):
    """
    Process all ]
    videos in a directory (supports person folders)
    
    Args:
        video_dir (str): Directory containing video files or person folders
        output_base_dir (str): Base directory for output frames
        frame_interval (int): Extract every nth frame
    
    Structure:
        video_dir/person1/video.mp4 -> output_base_dir/person1/video_name/frames
        video_dir/person2/video.mp4 -> output_base_dir/person2/video_name/frames
    """
    
    video_extensions = ('.mp4', '.avi', '.mov', '.flv', '.mkv')
    video_dir_path = Path(video_dir)
    
    # Check if videos are in person folders or directly in video_dir
    person_folders = [f for f in video_dir_path.iterdir() if f.is_dir()]
    
    if person_folders:
        # Process person folders (OpenGait structure)
        print(f"\nFound {len(person_folders)} person folder(s)")
        
        for person_folder in person_folders:
            person_name = person_folder.name
            print(f"\n{'='*60}")
            print(f"Processing Person: {person_name}")
            print(f"{'='*60}")
            
            # Find videos in person folder
            video_files = []
            for ext in video_extensions:
                video_files.extend(person_folder.glob(f'*{ext}'))
                video_files.extend(person_folder.glob(f'*{ext.upper()}'))
            
            if not video_files:
                print(f"  No videos found for {person_name}")
                continue
            
            print(f"  Found {len(video_files)} video(s)")
            
            for video_file in video_files:
                video_name = video_file.stem
                # Output: frames/person1/video_name/
                output_dir = os.path.join(output_base_dir, person_name, video_name)
                extract_frames(str(video_file), output_dir, frame_interval)
                print("-" * 50)
    else:
        # Process videos directly in video_dir (flat structure)
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir_path.glob(f'*{ext}'))
            video_files.extend(video_dir_path.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"No video files found in {video_dir}")
            return
        
        print(f"\nFound {len(video_files)} video file(s)")
        
        for video_file in video_files:
            video_name = video_file.stem
            output_dir = os.path.join(output_base_dir, video_name)
            extract_frames(str(video_file), output_dir, frame_interval)
            print("-" * 50)


if __name__ == "__main__":
    # Example usage
    video_dir = "../data/videos"
    output_dir = "../data/frames"
    frame_interval = 1  # Extract every frame (change to 2, 5, etc. for sampling)
    
    # Process all videos
    process_all_videos(video_dir, output_dir, frame_interval)
