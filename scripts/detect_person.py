"""
Person Detection Module
Detects and crops persons from frames using YOLOv11
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


class PersonDetector:
    """Detect and crop persons from frames using YOLO"""

    def __init__(self, model='yolo11n.pt', conf=0.5, device='cpu'):
       
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        
        self.conf = conf
        self.device = device
        
        print(f"Loading YOLO model: {model}")
        try:
            self.yolo = YOLO(model)
            print(f"✓ YOLO model loaded successfully")
            print(f"  Device: {device}")
            print(f"  Confidence threshold: {conf}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{model}': {e}")
    
    def detect_persons(self, frame):
        """
        Detect persons in a frame
        
        Args:
            frame (np.ndarray): Input frame (BGR)
        
        Returns:
            list: List of bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        # Run YOLO detection
        results = self.yolo(frame, conf=self.conf, device=self.device, verbose=False)
        
        persons = []
        
        if len(results) == 0:
            return persons
        
        result = results[0]
        
        # Extract person detections (class 0 in COCO)
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    persons.append((int(x1), int(y1), int(x2), int(y2), confidence))
        
        return persons
    
    def crop_person(self, frame, bbox, padding=10):
        """
        Crop person from frame with padding
        
        Args:
            frame (np.ndarray): Input frame
            bbox (tuple): Bounding box (x1, y1, x2, y2, conf)
            padding (int): Padding around bounding box
        
        Returns:
            np.ndarray: Cropped person image
        """
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
        """
        Draw bounding boxes on frame
        
        Args:
            frame (np.ndarray): Input frame
            persons (list): List of person detections
        
        Returns:
            np.ndarray: Frame with drawn boxes
        """
        frame_copy = frame.copy()
        
        for person in persons:
            x1, y1, x2, y2, conf = person
            
            # Draw rectangle
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person {conf:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy
    
    def process_folder(self, input_dir, output_dir, save_bbox_images=True, save_crops=True):
        """
        Process all frames in a folder
        
        Args:
            input_dir (str): Directory with input frames
            output_dir (str): Directory to save detected persons
            save_bbox_images (bool): Save images with bounding boxes drawn
            save_crops (bool): Save cropped person images
        
        Returns:
            int: Number of persons detected
        """
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
                
                pbar.update(1)
        
        print(f"\nDetection complete:")
        print(f"  Frames processed: {len(frame_files)}")
        print(f"  Frames with persons: {frames_with_persons}")
        print(f"  Total persons detected: {total_persons}")
        print(f"  Output saved to: {output_dir}")
        
        return total_persons


def process_all_frame_folders(frames_base_dir, output_base_dir, model='yolo11n.pt', conf=0.5, device='cpu', save_bbox_images=True, save_crops=True):

    frames_path = Path(frames_base_dir)
    
    # Check if we have person folders or direct video folders
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
    # Example usage
    frames_dir = "../data/frames"
    output_dir = "../data/detected_persons"
    
    # YOLO detection
    model = 'yolo11n.pt'  # Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
    conf = 0.5
    device = 'cpu'  # Use 'cuda' or '0' for GPU
    
    # Process all frame folders
    process_all_frame_folders(
        frames_dir, 
        output_dir, 
        model=model, 
        conf=conf, 
        device=device,
        save_bbox_images=True,
        save_crops=True
    )
