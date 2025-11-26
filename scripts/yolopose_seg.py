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
    results = seg_model(frame, verbose=False, conf=conf_threshold)
    silhouette = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    if results[0].masks is not None:
        for i, cls in enumerate(results[0].boxes.cls):
            if int(cls) == 0:  # Person class
                mask = results[0].masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                silhouette = np.maximum(silhouette, (mask_resized * 255).astype(np.uint8))
        
        if apply_morphology and silhouette.max() > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, kernel, iterations=2)
            silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return silhouette

def draw_skeleton(frame, keypoints, confidence_threshold=0.5, line_thickness=3, 
                 keypoint_radius=5):
    skeleton_parts = {
        'head': {
            'connections': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13]],
            'color': (255, 255, 0)
        },
        'torso': {
            'connections': [[6, 12], [7, 13], [6, 7]],
            'color': (0, 255, 255)
        },
        'left_arm': {
            'connections': [[6, 8], [8, 10]],
            'color': (255, 0, 0)
        },
        'right_arm': {
            'connections': [[7, 9], [9, 11]],
            'color': (0, 0, 255)
        },
        'left_leg': {
            'connections': [[12, 14], [14, 16]],
            'color': (255, 0, 255)
        },
        'right_leg': {
            'connections': [[13, 15], [15, 17]],
            'color': (0, 255, 0)
        }
    }
    
    point_color = (255, 0, 0)
    annotated = frame.copy()
    
    if len(keypoints.shape) == 3:
        for person_kpts in keypoints:
            for part_name, part_info in skeleton_parts.items():
                for connection in part_info['connections']:
                    pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
                    
                    if (person_kpts[pt1_idx][2] > confidence_threshold and 
                        person_kpts[pt2_idx][2] > confidence_threshold):
                        
                        pt1 = (int(person_kpts[pt1_idx][0]), int(person_kpts[pt1_idx][1]))
                        pt2 = (int(person_kpts[pt2_idx][0]), int(person_kpts[pt2_idx][1]))
                        
                        cv2.line(annotated, pt1, pt2, part_info['color'], line_thickness)
            
            for kpt in person_kpts:
                if kpt[2] > confidence_threshold:
                    center = (int(kpt[0]), int(kpt[1]))
                    cv2.circle(annotated, center, keypoint_radius, point_color, -1)
                    cv2.circle(annotated, center, keypoint_radius, (0, 0, 0), 1)
    
    return annotated

def get_person_bbox(keypoints, confidence_threshold=0.5, padding=50):
    valid_points = keypoints[keypoints[:, 2] > confidence_threshold]
    
    if len(valid_points) < 3:
        return None
    
    x_coords = valid_points[:, 0]
    y_coords = valid_points[:, 1]
    
    x1 = max(0, int(x_coords.min() - padding))
    y1 = max(0, int(y_coords.min() - padding))
    x2 = int(x_coords.max() + padding)
    y2 = int(y_coords.max() + padding)
    
    return (x1, y1, x2, y2)

def create_person_keypoints_csv(output_folder, all_keypoints_data):
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    print(f"\nCreating keypoint CSV files...")
    
    for person_id, frames_data in all_keypoints_data.items():
        csv_path = output_folder / f"{person_id}_keypoints.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            header = ['frame', 'frame_number']
            for name in keypoint_names:
                header.extend([f'x_{name}', f'y_{name}', f'conf_{name}'])
            
            writer.writerow(header)
            
            for frame_data in frames_data:
                frame_name = frame_data['frame_name']
                frame_num = frame_data['frame_number']
                keypoints = frame_data['keypoints']
                
                row = [frame_name, frame_num]
                
                for i in range(17):
                    x = keypoints[i][0]
                    y = keypoints[i][1]
                    conf = keypoints[i][2]
                    row.extend([f"{x:.2f}", f"{y:.2f}", f"{conf:.3f}"])
                
                writer.writerow(row)
        
        print(f"  Created: {person_id}_keypoints.csv ({len(frames_data)} frames)")

def process_single_video(video_path, pose_model, seg_model, output_folders, 
                        skip_frames=0, conf_threshold=0.5, line_thickness=3,
                        extraction_mode='both', apply_morphology=True, 
                        tracker_type='botsort.yaml'):
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
    print(f"Tracker: {tracker_type.replace('.yaml', '').upper()}")
    print(f"Extraction mode: {extraction_mode}")
    
    # Create window
    window_name = f"Pose Tracking - {Path(video_path).name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Counters
    saved_count = {'cropped': 0, 'skeleton': 0, 'silhouette': 0}
    
    # Store keypoints data for CSV export
    all_keypoints_data = {}
    
    # Track unique person IDs
    active_track_ids = set()
    
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
            
            # Perform pose detection WITH TRACKING
            # persist=True maintains IDs across frames0000000000000
            if tracker_type == "deepsort" :
                #call that function
            
            results = pose_model.track(
                frame, 
                persist=True,
                tracker=tracker_type,
                conf=conf_threshold,
                verbose=False
            )
            
            # Draw custom skeleton
            annotated_frame = frame.copy()
            
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                keypoints = results[0].keypoints.data.cpu().numpy()
                annotated_frame = draw_skeleton(frame, keypoints, conf_threshold, 
                                               line_thickness, 5)
                
                # Get tracking IDs (None if no tracking available)
                track_ids = None
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # Process each tracked person
                for person_idx, person_kpts in enumerate(keypoints):
                    bbox = get_person_bbox(person_kpts, conf_threshold)
                    
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        
                        # Ensure bbox is within frame
                        x2 = min(x2, width)
                        y2 = min(y2, height)
                        
                        if x2 > x1 and y2 > y1:
                            # Get consistent track ID
                            if track_ids is not None and person_idx < len(track_ids):
                                track_id = int(track_ids[person_idx])
                            else:
                                track_id = person_idx
                            
                            active_track_ids.add(track_id)
                            
                            # Generate consistent person ID
                            person_id = f"{video_name}_person{track_id:03d}"
                            
                            # Crop person from original frame
                            cropped_person = frame[y1:y2, x1:x2]
                            
                            # Save cropped person
                            cropped_filename = f"{video_name}_f{frame_count:06d}_p{track_id:03d}.jpg"
                            cropped_path = output_folders['cropped'] / cropped_filename
                            cv2.imwrite(str(cropped_path), cropped_person)
                            saved_count['cropped'] += 1
                            
                            # Store keypoints data for CSV
                            if person_id not in all_keypoints_data:
                                all_keypoints_data[person_id] = []
                            
                            all_keypoints_data[person_id].append({
                                'frame_name': cropped_filename,
                                'frame_number': frame_count,
                                'keypoints': person_kpts
                            })
                            
                            # Draw track ID on frame
                            label = f"ID:{track_id}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]   
                            
                            # Background for label
                            cv2.rectangle(annotated_frame, 
                                        (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), 
                                        (255, 255, 0), -1)
                            cv2.putText(annotated_frame, label, (x1, y1-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (143, 0, 255), 2)
                            
                            # Process based on extraction mode
                            if extraction_mode in ['skeleton', 'both']:
                                cropped_skeleton = annotated_frame[y1:y2, x1:x2]
                                skeleton_filename = f"{video_name}_f{frame_count:06d}_p{track_id:03d}_skeleton.jpg"
                                skeleton_path = output_folders['skeleton'] / skeleton_filename
                                cv2.imwrite(str(skeleton_path), cropped_skeleton)
                                saved_count['skeleton'] += 1
                            
                            if extraction_mode in ['silhouette', 'both'] and seg_model is not None:
                                silhouette_mask = extract_silhouette(cropped_person, seg_model, 
                                                                    conf_threshold, apply_morphology)
                                silhouette_img = cv2.cvtColor(silhouette_mask, cv2.COLOR_GRAY2BGR)
                                
                                silhouette_filename = f"{video_name}_f{frame_count:06d}_p{track_id:03d}_silhouette.jpg"
                                silhouette_path = output_folders['silhouette'] / silhouette_filename
                                cv2.imwrite(str(silhouette_path), silhouette_img)
                                saved_count['silhouette'] += 1
            
            # Create side-by-side display
            combined = np.hstack((frame, annotated_frame))
            
            # Add text labels
            cv2.putText(combined, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, f"Tracked ({tracker_type.split('.')[0].upper()})", 
                       (width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add tracking info
            status_text = f"Saved: {saved_count['cropped']} | Unique IDs: {len(active_track_ids)}"
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
    print(f"Total unique persons tracked: {len(active_track_ids)}")

def process_videos_with_pose_detection(videos_folder='videos', 
                                      model_path='yolo11n-pose.pt',
                                      seg_model_path='yolo11n-seg.pt',
                                      skip_frames=0, 
                                      conf_threshold=0.5, 
                                      line_thickness=3,
                                      extraction_mode='both',
                                      output_folder='data',
                                      apply_morphology=True,
                                      tracker_type='botsort.yaml'):
    
    print(f"Loading YOLO pose model: {model_path}")
    pose_model = YOLO(model_path)
    
    seg_model = None
    if extraction_mode in ['silhouette', 'both']:
        print(f"Loading YOLO segmentation model: {seg_model_path}")
        seg_model = YOLO(seg_model_path)
    
    output_folders = create_output_folders(output_folder)
    
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
    
    for video_file in video_files:
        print(f"\n{'='*60}")
        print(f"Processing: {video_file.name}")
        print(f"{'='*60}")
        process_single_video(str(video_file), pose_model, seg_model, output_folders,
                           skip_frames, conf_threshold, line_thickness, 
                           extraction_mode, apply_morphology, tracker_type)

def main():
    VIDEOS_FOLDER = 'videos'
    OUTPUT_FOLDER = 'data'
    
    POSE_MODEL_PATH = 'yolo11n-pose.pt'
    SEG_MODEL_PATH = 'yolo11n-seg.pt'
    
    # Options: 'botsort.yaml' (best accuracy) or 'bytetrack.yaml' (fastest)
    TRACKER_TYPE = 'botsort.yaml'
    
    SKIP_FRAMES = 3
    CONF_THRESHOLD = 0.5
    LINE_THICKNESS = 3
    EXTRACTION_MODE = 'both'
    APPLY_MORPHOLOGY = True
    
    print("="*60)
    print("YOLOv11 Pose Detection with Built-in Tracking")
    print("="*60)
    print(f"Tracker: {TRACKER_TYPE.replace('.yaml', '').upper()}")
    print(f"Extraction Mode: {EXTRACTION_MODE.upper()}")
    print(f"Output Folder: {OUTPUT_FOLDER}/")
    print("\nTracking Options:")
    print("  BoT-SORT: Best accuracy, handles occlusions & camera motion")
    print("  ByteTrack: Fastest, good for real-time & simple scenarios")
    print("  ESC - Stop processing")
    print("  P   - Pause/Resume")
    print("="*60)
    
    process_videos_with_pose_detection(
        videos_folder=VIDEOS_FOLDER,
        model_path=POSE_MODEL_PATH,
        seg_model_path=SEG_MODEL_PATH,
        skip_frames=SKIP_FRAMES,
        conf_threshold=CONF_THRESHOLD,
        line_thickness=LINE_THICKNESS,
        extraction_mode=EXTRACTION_MODE,
        output_folder=OUTPUT_FOLDER,
        apply_morphology=APPLY_MORPHOLOGY,
        tracker_type=TRACKER_TYPE
    )
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print(f"Check '{OUTPUT_FOLDER}/' folder for outputs")
    print("="*60)

if __name__ == "__main__":
    main()