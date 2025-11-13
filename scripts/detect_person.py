import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO
import csv


class PersonDetector:
    def __init__(self, model='yolo11n.pt', conf=0.3, device='cpu'):
        
        self.conf = conf
        self.device = device
        self.model_name = model
        self.is_pose_model = '-pose' in model.lower()
        
        print(f"Loading YOLO model: {model}")
        print(f"  Type: {'Pose estimation' if self.is_pose_model else 'Object detection'}")
        print(f"  Confidence threshold: {conf} (lower = detect more distant persons)")
        
        try:
            self.yolo = YOLO(model)
            print(f"✓ YOLO model loaded successfully")
            print(f"  Device: {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{model}': {e}")
    
    def detect_persons(self, frame):
        # Run YOLO detection with lower confidence to catch distant persons
        results = self.yolo(frame, conf=self.conf, device=self.device, verbose=False)
        
        persons = []
        if len(results) == 0:
            return persons
        
        result = results[0]
        
        # Extract all person detections (class 0 in COCO dataset)
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0])
                if cls == 0:  # Person class
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    # Filter out very small detections (likely false positives)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width < 20 or box_height < 40:  # Minimum person size
                        continue
                    
                    # Add keypoints if pose model
                    if self.is_pose_model and result.keypoints is not None:
                        if i < len(result.keypoints.xy):
                            keypoints = result.keypoints.xy[i].cpu().numpy()
                            keypoint_conf = result.keypoints.conf[i].cpu().numpy() if result.keypoints.conf is not None else None
                            persons.append((int(x1), int(y1), int(x2), int(y2), confidence, keypoints, keypoint_conf))
                        else:
                            persons.append((int(x1), int(y1), int(x2), int(y2), confidence, None, None))
                    else:
                        # Regular detection model
                        persons.append((int(x1), int(y1), int(x2), int(y2), confidence))
        
        return persons
    
    def crop_person(self, frame, bbox, padding=10):
        x1, y1, x2, y2 = bbox[:4]
        h, w = frame.shape[:2]
        
        # Add padding but keep within frame bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop person region
        cropped = frame[y1:y2, x1:x2]
        return cropped
    
    def draw_detections(self, frame, persons):
        frame_copy = frame.copy()
        
        # COCO pose skeleton connections
        pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),      # 0-3: Head connections
            (5, 6), (5, 11), (6, 12), (11, 12),  # 4-7: Torso connections
            (5, 7), (7, 9),                      # 8-9: Left Arm
            (6, 8), (8, 10),                     # 10-11: Right Arm
            (11, 13), (13, 15),                  # 12-13: Left Leg
            (12, 14), (14, 16)                   # 14-15: Right Leg
        ]
        colors = {
            "bright_green": (0, 255, 0),   # 🟢 Bounding Box default
            "orange":       (0, 165, 255),  # 🟠 Bounding Box for closest person (person 0)
            "red_kp":       (0, 0, 255),    # 🔴 Keypoint color
            "white_kp_out": (255, 255, 255),# ⚪ Keypoint outline color
            "yellow_head":  (0, 255, 255),  # 💛 Head (Nose, Eyes, Ears)
            "green_torso":  (0, 255, 0),    # 💚 Torso (Shoulders, Hips, Middle)
            "magenta_l_arm": (255, 0, 255), # 💜 Left Arm
            "blue_r_arm":   (255, 0, 0),    # 🔵 Right Arm
            "pink_l_leg":   (255, 0, 128),  # 💗 Left Leg (Using a slightly different pink/purple)
            "cyan_r_leg":   (255, 255, 0),  # 🔵 Light blue/Cyan Right Leg
        }
        
        # Assign colors to each connection
        connection_colors = (
            [colors["yellow_head"]] * 4 + 
            [colors["green_torso"]] * 4 + 
            [colors["magenta_l_arm"]] * 2 +
            [colors["blue_r_arm"]] * 2 +
            [colors["pink_l_leg"]] * 2 +
            [colors["cyan_r_leg"]] * 2
        )
        
        # Draw each detected person
        for person_idx, person in enumerate(persons):
            x1, y1, x2, y2, conf = person[:5]
            
            # Draw bounding box (green for all persons)
            # Closest person (index 0) is Orange, others are Bright Green
            box_color = colors["orange"] if person_idx == 0 else colors["bright_green"]
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw person label with ID
            label = f"Person{person_idx} {conf:.2f}"
            if self.is_pose_model:
                label = f"Person{person_idx}+Pose {conf:.2f}"
                
            # Label color matches box color
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Draw pose keypoints if available
            if self.is_pose_model and len(person) > 5:
                keypoints = person[5]
                keypoint_conf = person[6] if len(person) > 6 else None
                
                if keypoints is not None:
                    # Draw skeleton connections with colors
                    for idx, connection in enumerate(pose_connections):
                        pt1_idx, pt2_idx = connection
                        if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                            pt1 = keypoints[pt1_idx]
                            pt2 = keypoints[pt2_idx]
                            color = connection_colors[idx]
                            
                            # Only draw if both keypoints are confident
                            draw_line = True
                            if keypoint_conf is not None:
                                if keypoint_conf[pt1_idx] <= 0.5 or keypoint_conf[pt2_idx] <= 0.5:
                                    draw_line = False
                            
                            if draw_line:
                                cv2.line(frame_copy, (int(pt1[0]), int(pt1[1])),
                                         (int(pt2[0]), int(pt2[1])), color, 2)

                    # Draw keypoints as red circles with white outlines
                    for i, (x, y) in enumerate(keypoints):
                        if keypoint_conf is None or keypoint_conf[i] > 0.5:
                            # White outline (thicker circle)
                            cv2.circle(frame_copy, (int(x), int(y)), 4, colors["white_kp_out"], -1)
                            # Red center
                            cv2.circle(frame_copy, (int(x), int(y)), 2, colors["red_kp"], -1)
        
        return frame_copy
    
    
    def process_folder(self, input_dir, output_dir, save_bbox_images=True, save_crops=True):
        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if save_bbox_images:
            bbox_dir = os.path.join(output_dir, 'bbox_visualizations')
            Path(bbox_dir).mkdir(parents=True, exist_ok=True)
        
        if save_crops:
            crops_dir = os.path.join(output_dir, 'person_crops')
            Path(crops_dir).mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        frame_files = []
        
        for ext in image_extensions:
            frame_files.extend(sorted(Path(input_dir).glob(f'*{ext}')))
            frame_files.extend(sorted(Path(input_dir).glob(f'*{ext.upper()}')))
        
        if not frame_files:
            print(f"No frame files found in {input_dir}")
            return 0
        
        print(f"\nDetecting persons in {len(frame_files)} frames")
        print(f"Confidence threshold: {self.conf} (lower = detect more distant persons)")
        
        total_persons = 0
        frames_with_persons = 0
        
        # Store all keypoints data for CSV creation
        all_keypoints_data = {}  # person_id -> list of frame data
        
        # Process each frame
        with tqdm(total=len(frame_files), desc="Processing frames") as pbar:
            for frame_path in frame_files:
                frame = cv2.imread(str(frame_path))
                
                if frame is None:
                    pbar.update(1)
                    continue
                
                # Detect all persons in frame (including distant ones)
                persons = self.detect_persons(frame)
                
                if len(persons) > 0:
                    frames_with_persons += 1
                    total_persons += len(persons)
                    
                    # Save visualization with bounding boxes and poses
                    if save_bbox_images:
                        bbox_frame = self.draw_detections(frame, persons)
                        bbox_path = os.path.join(bbox_dir, frame_path.name)
                        cv2.imwrite(bbox_path, bbox_frame)
                    
                    # Save cropped person images and collect keypoints data
                    if save_crops:
                        for i, person in enumerate(persons):
                            # Crop person from frame
                            cropped = self.crop_person(frame, person)
                            crop_name = f"{frame_path.stem}_person{i}.jpg"
                            crop_path = os.path.join(crops_dir, crop_name)
                            cv2.imwrite(crop_path, cropped)
                            
                            # Collect keypoints data for CSV (if pose model)
                            if self.is_pose_model and len(person) > 5 and person[5] is not None:
                                person_id = f"person{i}"
                                
                                if person_id not in all_keypoints_data:
                                    all_keypoints_data[person_id] = []
                                
                                keypoints = person[5]
                                keypoint_conf = person[6] if len(person) > 6 else None
                                
                                # Store frame data
                                frame_data = {
                                    'frame_name': frame_path.stem,
                                    'keypoints': keypoints,
                                    'confidence': keypoint_conf
                                }
                                all_keypoints_data[person_id].append(frame_data)
                
                pbar.update(1)
        
        # Create single CSV file per person (if pose model)
        if self.is_pose_model and save_crops and all_keypoints_data:
            self._create_person_keypoints_csv(crops_dir, all_keypoints_data)
        
        # Print summary
        print(f"\nDetection Summary:")
        print(f"  Frames processed: {len(frame_files)}")
        print(f"  Frames with persons: {frames_with_persons}")
        print(f"  Total persons detected: {total_persons}")
        print(f"  Average persons per frame: {total_persons/len(frame_files):.2f}")
        print(f"  Output saved to: {output_dir}")
        if self.is_pose_model and all_keypoints_data:
            print(f"  Keypoints saved as: {len(all_keypoints_data)} person CSV files (Wide Format)")
        
        return total_persons
    
    def _create_person_keypoints_csv(self, crops_dir, all_keypoints_data):
        
        # COCO 17 keypoint names
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Create one WIDE FORMAT CSV file per person
        for person_id, frames_data in all_keypoints_data.items():
            csv_path = os.path.join(crops_dir, f"{person_id}_keypoints.csv")
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # --- WIDE FORMAT HEADER CREATION ---
                # Header will be: frame, x_nose, y_nose, conf_nose, x_left_eye, y_left_eye, conf_left_eye, ...
                header = ['frame']
                for name in keypoint_names:
                    header.append(f'x_{name}')
                    header.append(f'y_{name}')
                    header.append(f'conf_{name}')
                
                writer.writerow(header)
                # -----------------------------------
                
                # Write data for each frame
                for frame_data in frames_data:
                    frame_name = frame_data['frame_name']
                    keypoints = frame_data['keypoints']  # N x 2 array of (x, y)
                    keypoint_conf = frame_data['confidence']  # N array of confidence
                    
                    row = [frame_name]
                    
                    # --- WIDE FORMAT ROW DATA CREATION ---
                    for i, (x, y) in enumerate(keypoints):
                        conf = keypoint_conf[i] if keypoint_conf is not None else 1.0
                        
                        # Append x, y, and confidence for the keypoint
                        row.append(f"{x:.2f}")
                        row.append(f"{y:.2f}")
                        row.append(f"{conf:.3f}")
                    
                    writer.writerow(row)
                    # -------------------------------------
            
            print(f"  {person_id} keypoints (Wide Format): {csv_path} ({len(frames_data)} frames)")


def process_all_frame_folders(frames_base_dir, output_base_dir, model='yolo11n.pt', conf=0.3, device='cpu', save_bbox_images=True, save_crops=True):

    frames_path = Path(frames_base_dir)
    
    # Find all folders
    first_level_folders = [f for f in frames_path.iterdir() if f.is_dir()]
    
    if not first_level_folders:
        print(f"No folders found in {frames_base_dir}")
        return
    
    # Check folder structure (person folders vs direct video folders)
    has_nested_structure = any(
        any(subfolder.is_dir() for subfolder in folder.iterdir())
        for folder in first_level_folders
    )
    
    # Initialize detector once (reuse model for efficiency)
    print(f"\nInitializing person detector...")
    print(f"Model: {model}")
    detector = PersonDetector(model=model, conf=conf, device=device)
    
    if has_nested_structure:
        # Process person-based folder structure: data/frames/person1/video1/
        print(f"\nProcessing person-based folder structure")
        
        for person_folder in first_level_folders:
            if not person_folder.is_dir():
                continue
            
            person_name = person_folder.name
            print(f"\n{'='*60}")
            print(f"Processing Person: {person_name}")
            print(f"{'='*60}")
            
            # Process each video folder for this person
            video_folders = [f for f in person_folder.iterdir() if f.is_dir()]
            
            for video_folder in video_folders:
                video_name = video_folder.name
                print(f"\n  Video: {video_name}")
                print(f"  {'-'*50}")
                
                output_dir = os.path.join(output_base_dir, person_name, video_name)
                detector.process_folder(str(video_folder), output_dir, save_bbox_images, save_crops)
    else:
        # Process flat structure: data/frames/video1/
        print(f"\nProcessing flat folder structure")
        
        for video_folder in first_level_folders:
            if video_folder.is_dir():
                print(f"\n{'='*50}")
                print(f"Processing: {video_folder.name}")
                print(f"{'='*50}")
                
                output_dir = os.path.join(output_base_dir, video_folder.name)
                detector.process_folder(str(video_folder), output_dir, save_bbox_images, save_crops)


if __name__ == "__main__":
    # Configuration - CHANGE HERE FOR DIFFERENT SETTINGS
    frames_dir = "../data/frames"
    output_dir = "../data/detected_persons"
    
    model = 'yolo11n-pose.pt'  # Options: yolo11n-pose.pt, yolo11s-pose.pt, yolo11m-pose.pt

    conf = 0.3  # Lower confidence to detect distant persons (0.1-0.9)
    device = 'cpu'  # Use 'cuda' for GPU acceleration
    
    # Run detection on all frame folders
    process_all_frame_folders(
        frames_dir, 
        output_dir, 
        model=model, 
        conf=conf, 
        device=device,
        save_bbox_images=True,  # Save visualization images
        save_crops=True         # Save cropped person images and keypoint CSVs
    )