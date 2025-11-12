"""
Person Detection Module
Detects and crops persons from frames using YOLOv11
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model='yolo11n.pt', conf=0.5, device='cpu'):
        
        self.conf = conf
        self.device = device
        self.model_name = model
        
        # Check if it's a pose model
        self.is_pose_model = '-pose' in model.lower()
        
        print(f"Loading YOLO model: {model}")
        if self.is_pose_model:
            print(f"  Type: Pose estimation (detection + keypoints)")
        else:
            print(f"  Type: Object detection")
        
        try:
            self.yolo = YOLO(model)
            print(f"✓ YOLO model loaded successfully")
            print(f"  Device: {device}")
            print(f"  Confidence threshold: {conf}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{model}': {e}")
    
    def detect_persons(self, frame):

        # Run YOLO detection/pose estimation
        results = self.yolo(frame, conf=self.conf, device=self.device, verbose=False)
        
        persons = []
        
        if len(results) == 0:
            return persons
        
        result = results[0]
        
        # Extract person detections (class 0 in COCO)
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0])
                if cls == 0:  # Person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    # Add keypoints if pose model
                    if self.is_pose_model and result.keypoints is not None:
                        if i < len(result.keypoints.xy):
                            keypoints = result.keypoints.xy[i].cpu().numpy()  # Shape: (17, 2)
                            keypoint_conf = result.keypoints.conf[i].cpu().numpy() if result.keypoints.conf is not None else None
                            persons.append((int(x1), int(y1), int(x2), int(y2), confidence, keypoints, keypoint_conf))
                        else:
                            persons.append((int(x1), int(y1), int(x2), int(y2), confidence, None, None))
                    else:
                        persons.append((int(x1), int(y1), int(x2), int(y2), confidence))
        
        return persons
    
    def crop_person(self, frame, bbox, padding=10):

        x1, y1, x2, y2 = bbox[:4]
        h, w = frame.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop
        cropped = frame[y1:y2, x1:x2]
        
        return cropped
    
    def draw_detections(self, frame, persons):

        frame_copy = frame.copy()
        
        # COCO pose keypoint connections (skeleton)
        pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        colors ={
            "head": (0, 255, 255),
            "arms": (255, 0, 255),  
            "torso": (0, 255, 0),
            "legs": (255, 0, 0)
        }
        connection_colors = (
        [colors["head"]] * 4 +
        [colors["arms"]] * 5 +
        [colors["torso"]] * 3 +
        [colors["legs"]] * 4
    )
        for person in persons:
            x1, y1, x2, y2, conf = person[:5]
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person+Pose {conf:.2f}" if self.is_pose_model else f"Person {conf:.2f}"
            cv2.putText(
            frame_copy, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            
            # Draw pose keypoints if available
            if self.is_pose_model and len(person) > 5:
             keypoints = person[5]
            keypoint_conf = person[6] if len(person) > 6 else None
            
            if keypoints is not None:
                # Draw keypoints
                
                for i, (x, y) in enumerate(keypoints):
                    if keypoint_conf is None or keypoint_conf[i] > 0.5:
                        cv2.circle(frame_copy, (int(x), int(y)), 3, (0, 0, 255), -1)
                        cv2.putText(
                            frame_copy, str(i), (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                        )

                
                # Draw skeleton with multiple colors
                for idx, connection in enumerate(pose_connections):
                    pt1_idx, pt2_idx = connection
                    if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                        pt1 = keypoints[pt1_idx]
                        pt2 = keypoints[pt2_idx]
                        color = connection_colors[idx]
                        
                        if keypoint_conf is not None:
                            if keypoint_conf[pt1_idx] > 0.5 and keypoint_conf[pt2_idx] > 0.5:
                                cv2.line(frame_copy, (int(pt1[0]), int(pt1[1])),
                                         (int(pt2[0]), int(pt2[1])), color, 2)
                        else:
                            cv2.line(frame_copy, (int(pt1[0]), int(pt1[1])),
                                     (int(pt2[0]), int(pt2[1])), color, 2)
        
        return frame_copy
    
    def process_folder(self, input_dir, output_dir, save_bbox_images=True, save_crops=True):

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
        
        total_persons = 0
        frames_with_persons = 0
        
        with tqdm(total=len(frame_files)) as pbar:
            for frame_path in frame_files:
                frame = cv2.imread(str(frame_path))
                
                if frame is None:
                    pbar.update(1)
                    continue
                
                # Detect persons
                persons = self.detect_persons(frame)
                
                if len(persons) > 0:
                    frames_with_persons += 1
                    total_persons += len(persons)
                    
                    # Save bbox visualization
                    if save_bbox_images:
                        bbox_frame = self.draw_detections(frame, persons)
                        bbox_path = os.path.join(bbox_dir, frame_path.name)
                        cv2.imwrite(bbox_path, bbox_frame)
                    
                    # Save cropped persons
                    if save_crops:
                        for i, person in enumerate(persons):
                            cropped = self.crop_person(frame, person)
                            crop_name = f"{frame_path.stem}_person{i}.jpg"
                            crop_path = os.path.join(crops_dir, crop_name)
                            cv2.imwrite(crop_path, cropped)
                            
                            # Save keypoints data if pose model
                            if self.is_pose_model and len(person) > 5 and person[5] is not None:
                                keypoints_name = f"{frame_path.stem}_person{i}_keypoints.txt"
                                keypoints_path = os.path.join(crops_dir, keypoints_name)
                                
                                keypoints = person[5]
                                keypoint_conf = person[6] if len(person) > 6 else None
                                
                                with open(keypoints_path, 'w') as f:
                                    f.write("# COCO 17 keypoints format\n")
                                    f.write("# 0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear\n")
                                    f.write("# 5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow\n")
                                    f.write("# 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip\n")
                                    f.write("# 13:left_knee 14:right_knee 15:left_ankle 16:right_ankle\n")
                                    f.write("# Format: x y confidence\n")
                                    
                                    for j, (x, y) in enumerate(keypoints):
                                        conf = keypoint_conf[j] if keypoint_conf is not None else 1.0
                                        f.write(f"{x:.2f} {y:.2f} {conf:.3f}\n")
                
                pbar.update(1)
        
        print(f"\nDetection complete:")
        print(f"  Frames processed: {len(frame_files)}")
        print(f"  Frames with persons: {frames_with_persons}")
        print(f"  Total persons detected: {total_persons}")
        print(f"  Output saved to: {output_dir}")
        
        return total_persons


def process_all_frame_folders(frames_base_dir, output_base_dir, model='yolo11n.pt', conf=0.5, device='cpu', save_bbox_images=True, save_crops=True):

    frames_path = Path(frames_base_dir)
    
    first_level_folders = [f for f in frames_path.iterdir() if f.is_dir()]
    
    if not first_level_folders:
        print(f"No folders found in {frames_base_dir}")
        return
    
    # Check if first level contains person folders (by checking for nested folders)
    has_nested_structure = any(
        any(subfolder.is_dir() for subfolder in folder.iterdir())
        for folder in first_level_folders
    )
    
    # Create single detector instance (reuse YOLO model)
    print(f"\nInitializing person detector...")
    print(f"Model: {model}")
    detector = PersonDetector(model=model, conf=conf, device=device)
    
    if has_nested_structure:
        # Process person folders (OpenGait structure)
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
        # Process flat structure (direct video folders)
        print(f"\nProcessing flat folder structure")
        
        for video_folder in first_level_folders:
            if video_folder.is_dir():
                print(f"\n{'='*50}")
                print(f"Processing: {video_folder.name}")
                print(f"{'='*50}")
                
                output_dir = os.path.join(output_base_dir, video_folder.name)
                detector.process_folder(str(video_folder), output_dir, save_bbox_images, save_crops)


if __name__ == "__main__":

    frames_dir = "../data/frames"
    output_dir = "../data/detected_persons"
    
    
    model = 'yolo11n-pose.pt' 
    
    conf = 0.5
    device = 'cpu'  # Use 'cuda' or '0' for GPU
    

    process_all_frame_folders(
        frames_dir, 
        output_dir, 
        model=model, 
        conf=conf, 
        device=device,
        save_bbox_images=True,
        save_crops=True
    )
