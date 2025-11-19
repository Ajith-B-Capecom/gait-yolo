import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO
import csv

def create_output_folders(base_folder='data'):
    """Create necessary output folders."""
    folders = {
        'cropped': Path(base_folder) / 'cropped_persons',
        'skeleton': Path(base_folder) / 'skeleton_images',
        'silhouette': Path(base_folder) / 'silhouette_images',
        'csv': Path(base_folder) / 'keypoints_csv'
    }
    
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    return folders

def extract_silhouette(frame, seg_model, conf_threshold=0.5, apply_morphology=True):

    # Run segmentation
    results = seg_model(frame, verbose=False, conf=conf_threshold)
    
    # Create empty mask
    silhouette = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Extract person masks (class 0 is person in COCO)
    if results[0].masks is not None:
        for i, cls in enumerate(results[0].boxes.cls):
            if int(cls) == 0:  # Person class
                mask = results[0].masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                silhouette = np.maximum(silhouette, (mask_resized * 255).astype(np.uint8))
        
        # Apply morphological operations to clean up the silhouette
        if apply_morphology and silhouette.max() > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # Remove noise (small isolated pixels)
            silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, kernel, iterations=2)
            # Fill holes inside the silhouette
            silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return silhouette

def draw_skeleton(frame, keypoints, confidence_threshold=0.5, line_thickness=3, 
                 keypoint_radius=5):
    skeleton_parts = {
        'head': {
            'connections': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13]],
            'color': (255, 255, 0)  # Cyan
        },
        'torso': {
            'connections': [[6, 12], [7, 13], [6, 7]],
            'color': (0, 255, 255)  # Yellow
        },
        'left_arm': {
            'connections': [[6, 8], [8, 10]],
            'color': (255, 0, 0)  # Blue
        },
        'right_arm': {
            'connections': [[7, 9], [9, 11]],
            'color': (0, 0, 255)  # Red
        },
        'left_leg': {
            'connections': [[12, 14], [14, 16]],
            'color': (255, 0, 255)  # Magenta
        },
        'right_leg': {
            'connections': [[13, 15], [15, 17]],
            'color': (0, 255, 0)  # Green
        }
    }
    
    point_color = (255, 0, 0)  # Blue keypoints
    
    annotated = frame.copy()
    
    if len(keypoints.shape) == 3:
        for person_kpts in keypoints:
            # Draw lines with different colors for each body part
            for part_name, part_info in skeleton_parts.items():
                for connection in part_info['connections']:
                    pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
                    
                    if (person_kpts[pt1_idx][2] > confidence_threshold and 
                        person_kpts[pt2_idx][2] > confidence_threshold):
                        
                        pt1 = (int(person_kpts[pt1_idx][0]), int(person_kpts[pt1_idx][1]))
                        pt2 = (int(person_kpts[pt2_idx][0]), int(person_kpts[pt2_idx][1]))
                        
                        cv2.line(annotated, pt1, pt2, part_info['color'], line_thickness)
            
            # Draw keypoints
            for kpt in person_kpts:
                if kpt[2] > confidence_threshold:
                    center = (int(kpt[0]), int(kpt[1]))
                    cv2.circle(annotated, center, keypoint_radius, point_color, -1)
                    cv2.circle(annotated, center, keypoint_radius, (0, 0, 0), 1)
    
    return annotated

def get_person_bbox(keypoints, confidence_threshold=0.5, padding=20):
    """
    Calculate bounding box around detected person keypoints.
    """
    valid_points = keypoints[keypoints[:, 2] > confidence_threshold]
    
    if len(valid_points) < 3:  # Need at least 3 keypoints
        return None
    
    x_coords = valid_points[:, 0]
    y_coords = valid_points[:, 1]
    
    x1 = max(0, int(x_coords.min() - padding))
    y1 = max(0, int(y_coords.min() - padding))
    x2 = int(x_coords.max() + padding)
    y2 = int(y_coords.max() + padding)
    
    return (x1, y1, x2, y2)

def create_person_keypoints_csv(output_folder, all_keypoints_data):

    # COCO 17 keypoint names
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    print(f"\nCreating keypoint CSV files...")
    
    # Create one CSV file per person (wide format)
    for person_id, frames_data in all_keypoints_data.items():
        csv_path = output_folder / f"{person_id}_keypoints.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Create wide format header
            # Header: frame, x_nose, y_nose, conf_nose, x_left_eye, y_left_eye, conf_left_eye, ...
            header = ['frame', 'frame_number']
            for name in keypoint_names:
                header.extend([f'x_{name}', f'y_{name}', f'conf_{name}'])
            
            writer.writerow(header)
            
            # Write data for each frame
            for frame_data in frames_data:
                frame_name = frame_data['frame_name']
                frame_num = frame_data['frame_number']
                keypoints = frame_data['keypoints']  # shape: (17, 3) - (x, y, confidence)
                
                row = [frame_name, frame_num]
                
                # Add x, y, confidence for each of the 17 keypoints
                for i in range(17):
                    x = keypoints[i][0]
                    y = keypoints[i][1]
                    conf = keypoints[i][2]
                    row.extend([f"{x:.2f}", f"{y:.2f}", f"{conf:.3f}"])
                
                writer.writerow(row)
        
        print(f"  Created: {person_id}_keypoints.csv ({len(frames_data)} frames)")

def process_single_video(video_path, pose_model, seg_model, output_folders, 
                        skip_frames=0, conf_threshold=0.5, line_thickness=3,
                        extraction_mode='both', apply_morphology=True):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_name = Path(video_path).stem
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Extraction mode: {extraction_mode}")
    print(f"Silhouette morphology: {'Enabled' if apply_morphology else 'Disabled'}")
    if skip_frames > 0:
        print(f"Processing every {skip_frames + 1} frame(s) for speed")
    
    # Create window
    window_name = f"Pose Detection - {Path(video_path).name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Counters
    saved_count = {'cropped': 0, 'skeleton': 0, 'silhouette': 0}
    
    # Store keypoints data for CSV export
    # Format: {person_id: [frame_data_list]}
    all_keypoints_data = {}
    
    # Process frames
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames for faster processing
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                pbar.update(1)
                continue
            
            # Perform pose detection
            results = pose_model(frame, verbose=False, conf=conf_threshold, 
                               device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
            
            # Draw custom skeleton
            annotated_frame = frame.copy()
            
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                keypoints = results[0].keypoints.data.cpu().numpy()
                annotated_frame = draw_skeleton(frame, keypoints, conf_threshold, 
                                               line_thickness, 5)
                
                # Extract and save each detected person
                for person_idx, person_kpts in enumerate(keypoints):
                    bbox = get_person_bbox(person_kpts, conf_threshold)
                    
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        
                        # Ensure bbox is within frame
                        x2 = min(x2, width)
                        y2 = min(y2, height)
                        
                        if x2 > x1 and y2 > y1:
                            # Generate person ID
                            person_id = f"{video_name}_person{person_idx}"
                            
                            # Crop person from original frame
                            cropped_person = frame[y1:y2, x1:x2]
                            
                            # Save cropped person
                            cropped_filename = f"{video_name}_f{frame_count:06d}_p{person_idx}.jpg"
                            cropped_path = output_folders['cropped'] / cropped_filename
                            cv2.imwrite(str(cropped_path), cropped_person)
                            saved_count['cropped'] += 1
                            
                            # Store keypoints data for CSV
                            if person_id not in all_keypoints_data:
                                all_keypoints_data[person_id] = []
                            
                            all_keypoints_data[person_id].append({
                                'frame_name': cropped_filename,
                                'frame_number': frame_count,
                                'keypoints': person_kpts  # (17, 3) array with x, y, confidence
                            })
                            
                            # Process based on extraction mode
                            if extraction_mode in ['skeleton', 'both']:
                                # Crop skeleton image
                                cropped_skeleton = annotated_frame[y1:y2, x1:x2]
                                skeleton_filename = f"{video_name}_f{frame_count:06d}_p{person_idx}_skeleton.jpg"
                                skeleton_path = output_folders['skeleton'] / skeleton_filename
                                cv2.imwrite(str(skeleton_path), cropped_skeleton)
                                saved_count['skeleton'] += 1
                            
                            if extraction_mode in ['silhouette', 'both'] and seg_model is not None:
                                # Extract silhouette from cropped person
                                silhouette_mask = extract_silhouette(cropped_person, seg_model, 
                                                                    conf_threshold, apply_morphology)
                                
                                # Create 3-channel silhouette image
                                silhouette_img = cv2.cvtColor(silhouette_mask, cv2.COLOR_GRAY2BGR)
                                
                                silhouette_filename = f"{video_name}_f{frame_count:06d}_p{person_idx}_silhouette.jpg"
                                silhouette_path = output_folders['silhouette'] / silhouette_filename
                                cv2.imwrite(str(silhouette_path), silhouette_img)
                                saved_count['silhouette'] += 1
            
            # Create side-by-side display
            combined = np.hstack((frame, annotated_frame))
            
            # Add text labels
            cv2.putText(combined, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Skeleton Detection", (width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add extraction count
            status_text = f"Saved: {saved_count['cropped']} persons"
            cv2.putText(combined, status_text, (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(window_name, combined)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("\nStopped by user")
                break
            elif key == ord('p'):  # Pause on 'p' key
                cv2.waitKey(0)
            
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Create CSV files with keypoint data
    if all_keypoints_data:
        create_person_keypoints_csv(output_folders['csv'], all_keypoints_data)
    
    print(f"\nCompleted processing {frame_count} frames")
    print(f"Saved: {saved_count['cropped']} cropped persons, "
          f"{saved_count['skeleton']} skeleton images, "
          f"{saved_count['silhouette']} silhouette images")

def process_videos_with_pose_detection(videos_folder='videos', 
                                      model_path='yolo11n-pose.pt',
                                      seg_model_path='yolo11n-seg.pt',
                                      skip_frames=0, 
                                      conf_threshold=0.5, 
                                      line_thickness=3,
                                      extraction_mode='both',
                                      output_folder='data',
                                      apply_morphology=True):

    # Load models
    print(f"Loading YOLO pose model: {model_path}")
    pose_model = YOLO(model_path)
    
    seg_model = None
    if extraction_mode in ['silhouette', 'both']:
        print(f"Loading YOLO segmentation model: {seg_model_path}")
        seg_model = YOLO(seg_model_path)
    
    # Create output folders
    output_folders = create_output_folders(output_folder)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    videos_path = Path(videos_folder)
    
    if not videos_path.exists():
        print(f"Creating {videos_folder} folder...")
        videos_path.mkdir(parents=True, exist_ok=True)
        print(f"Please add video files to the '{videos_folder}' folder")
        return
    
    video_files = [f for f in videos_path.iterdir() 
                   if f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"No video files found in '{videos_folder}' folder")
        return
    
    print(f"Found {len(video_files)} video(s)")
    
    # Process each video
    for video_file in video_files:
        print(f"\n{'='*60}")
        print(f"Processing: {video_file.name}")
        print(f"{'='*60}")
        process_single_video(str(video_file), pose_model, seg_model, output_folders,
                           skip_frames, conf_threshold, line_thickness, 
                           extraction_mode, apply_morphology)

def main():

    VIDEOS_FOLDER = 'videos'
    OUTPUT_FOLDER = 'data'
    
    POSE_MODEL_PATH = 'yolo11n-pose.pt'  # yolo11n, s, m, l, x-pose.pt
    SEG_MODEL_PATH = 'yolo11n-seg.pt'    # yolo11n, s, m, l, x-seg.pt
    
    # SPEED OPTIMIZATION
    SKIP_FRAMES = 0  # 0=all frames, 1=skip 1 (2x faster), 2=skip 2 (3x faster)
    
    # DETECTION SETTINGS
    CONF_THRESHOLD = 0.5  # Confidence threshold (0.0-1.0)
    
    # SKELETON APPEARANCE
    LINE_THICKNESS = 4  # Thickness of skeleton lines

    # Options: 'skeleton', 'silhouette', 'both'
    EXTRACTION_MODE = 'both'
    
    # SILHOUETTE SETTINGS
    APPLY_MORPHOLOGY = True  # Apply morphological operations to clean silhouette
    
    # ===================================
    
    print("="*60)
    print("YOLOv11 Video Pose Detection with CSV Export")
    print("="*60)
    print(f"Extraction Mode: {EXTRACTION_MODE.upper()}")
    print(f"Output Folder: {OUTPUT_FOLDER}/")
    print(f"Silhouette Morphology: {'ENABLED' if APPLY_MORPHOLOGY else 'DISABLED'}")
    print("\nFeatures:")
    print("  - Detect and crop persons from video")
    print("  - Generate skeleton images")
    print("  - Generate silhouette masks")
    print("  - Export keypoints to CSV (wide format)")
    print("\nPerformance Tips:")
    print("  - Use smaller models for speed")
    print("  - Increase SKIP_FRAMES for faster processing")
    print("  - Use GPU if available (auto-detected)")
    print("\nControls:")
    print("  ESC - Stop processing")
    print("  P   - Pause/Resume")
    print("="*60)
    
    # Process videos
    process_videos_with_pose_detection(
        videos_folder=VIDEOS_FOLDER,
        model_path=POSE_MODEL_PATH,
        seg_model_path=SEG_MODEL_PATH,
        skip_frames=SKIP_FRAMES,
        conf_threshold=CONF_THRESHOLD,
        line_thickness=LINE_THICKNESS,
        extraction_mode=EXTRACTION_MODE,
        output_folder=OUTPUT_FOLDER,
        apply_morphology=APPLY_MORPHOLOGY
    )
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print(f"Check '{OUTPUT_FOLDER}/' folder for outputs:")
    print(f"  - {OUTPUT_FOLDER}/cropped_persons/ - Cropped person images")
    print(f"  - {OUTPUT_FOLDER}/skeleton_images/ - Skeleton overlay images")
    print(f"  - {OUTPUT_FOLDER}/silhouette_images/ - Silhouette masks")
    print(f"  - {OUTPUT_FOLDER}/keypoints_csv/ - Keypoint CSV files")
    print("="*60)

if __name__ == "__main__":
    main()