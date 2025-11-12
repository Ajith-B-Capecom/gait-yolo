"""
Silhouette Extraction Module
Extracts silhouettes from frames using YOLOv11 segmentation for better performance
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


class SilhouetteExtractor:

    def __init__(self, method='yolo', model='yolo11n-seg.pt', conf=0.25, device='cpu'):
        """
        Initialize silhouette extractor
        
        Args:
            method (str): Extraction method ('yolo', 'mog2', 'knn')
            model (str): YOLO model name (for yolo method)
                        - yolo11n-seg.pt (nano, fastest)
                        - yolo11s-seg.pt (small)
                        - yolo11m-seg.pt (medium)
                        - yolo11l-seg.pt (large)
                        - yolo11x-seg.pt (extra large, most accurate)
            conf (float): Confidence threshold for YOLO detection
            device (str): Device to run on ('cpu', 'cuda', '0', '1', etc.)
        """
        self.method = method
        self.conf = conf
        self.device = device
        self.yolo = None
        self.backSub = None
        
        if method == 'yolo':
            if not YOLO_AVAILABLE:
                raise ImportError(
                    "ultralytics not installed. Install with: pip install ultralytics"
                )
            
            print(f"Loading YOLO model: {model}")
            try:
                self.yolo = YOLO(model)
                print(f"✓ YOLO model loaded successfully")
                print(f"  Device: {device}")
                print(f"  Confidence threshold: {conf}")
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO model '{model}': {e}")
        
        elif method == 'mog2':
            self.backSub = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
        
        elif method == 'knn':
            self.backSub = cv2.createBackgroundSubtractorKNN(
                detectShadows=True,
                history=500
            )
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'yolo', 'mog2', or 'knn'")
    
    def get_silhouette_yolo(self, frame):
        """
        Extract silhouette using YOLO segmentation
        
        Args:
            frame (np.ndarray): Input frame (BGR)
        
        Returns:
            np.ndarray: Binary silhouette mask (0 or 255)
        """
        if self.yolo is None:
            raise RuntimeError("YOLO model not initialized")
        
        # Run YOLO segmentation
        results = self.yolo(frame, conf=self.conf, device=self.device, verbose=False)
        
        # Create empty mask
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(results) == 0:
            return mask
        
        result = results[0]
        
        # Check if segmentation masks are available
        if result.masks is None:
            return mask
        
        # Extract person masks (class 0 in COCO)
        if result.boxes is not None:
            for i, cls in enumerate(result.boxes.cls):
                if int(cls) == 0:  # Person class
                    # Get segmentation mask
                    if i < len(result.masks.data):
                        seg_mask = result.masks.data[i].cpu().numpy()
                        
                        # Resize mask to original image size
                        seg_mask = cv2.resize(seg_mask, (w, h))
                        
                        # Convert to binary (0 or 255)
                        binary_mask = (seg_mask > 0.5).astype(np.uint8) * 255
                        
                        # Combine with existing mask (OR operation)
                        mask = cv2.bitwise_or(mask, binary_mask)
        
        return mask
    
    def get_silhouette_background_subtraction(self, frame, apply_morphology=True):
        """
        Extract silhouette using background subtraction
        
        Args:
            frame (np.ndarray): Input frame
            apply_morphology (bool): Apply morphological operations
        
        Returns:
            np.ndarray: Binary silhouette mask
        """
        if self.backSub is None:
            raise RuntimeError("Background subtractor not initialized")
        
        # Apply background subtraction
        mask = self.backSub.apply(frame)
        
        if apply_morphology:
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # Remove noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Fill holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask
    
    def get_silhouette(self, frame, apply_morphology=True):
        """
        Extract silhouette from a frame using configured method
        
        Args:
            frame (np.ndarray): Input frame
            apply_morphology (bool): Apply morphological operations (for background subtraction)
        
        Returns:
            np.ndarray: Binary silhouette mask
        """
        if self.method == 'yolo':
            return self.get_silhouette_yolo(frame)
        else:
            return self.get_silhouette_background_subtraction(frame, apply_morphology)
    
    def extract_silhouettes_from_folder(self, input_dir, output_dir):
        """
        Extract silhouettes from all frames in a folder
        
        Args:
            input_dir (str): Directory with input frames
            output_dir (str): Directory to save silhouettes
        
        Returns:
            int: Number of silhouettes extracted
        """
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        frame_files = []
        
        for ext in image_extensions:
            frame_files.extend(sorted(Path(input_dir).glob(f'*{ext}')))
            frame_files.extend(sorted(Path(input_dir).glob(f'*{ext.upper()}')))
        
        if not frame_files:
            print(f"No frame files found in {input_dir}")
            return 0
        
        print(f"\nExtracting silhouettes from {len(frame_files)} frames")
        print(f"Method: {self.method}")
        
        count = 0
        with tqdm(total=len(frame_files)) as pbar:
            for frame_path in frame_files:
                frame = cv2.imread(str(frame_path))
                
                if frame is None:
                    pbar.update(1)
                    continue
                
                silhouette = self.get_silhouette(frame)
                
                output_name = frame_path.stem + '_silhouette.jpg'
                output_path = os.path.join(output_dir, output_name)
                
                cv2.imwrite(output_path, silhouette)
                count += 1
                pbar.update(1)
        
        print(f"Silhouettes saved to: {output_dir}")
        return count


def process_all_frame_folders(frames_base_dir, output_base_dir, method='yolo', model='yolo11n-seg.pt', conf=0.25, device='cpu'):
    """
    Process all frame folders to extract silhouettes (supports person folders)
    
    Args:
        frames_base_dir (str): Base directory containing frame subfolders
        output_base_dir (str): Base directory for output silhouettes
        method (str): Extraction method ('yolo', 'mog2', 'knn')
        model (str): YOLO model name (only for yolo method)
        conf (float): Confidence threshold (only for yolo method)
        device (str): Device to run on ('cpu', 'cuda', '0', '1', etc.)
    
    Structure:
        frames/person1/video_name/ -> silhouettes/person1/video_name/
        frames/person2/video_name/ -> silhouettes/person2/video_name/
    """
    
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
    
    # Create single extractor instance (reuse YOLO model)
    print(f"\nInitializing silhouette extractor...")
    print(f"Method: {method}")
    extractor = SilhouetteExtractor(method=method, model=model, conf=conf, device=device)
    
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
                extractor.extract_silhouettes_from_folder(str(video_folder), output_dir)
    else:
        # Process flat structure (direct video folders)
        print(f"\nProcessing flat folder structure")
        
        for video_folder in first_level_folders:
            if video_folder.is_dir():
                print(f"\n{'='*50}")
                print(f"Processing: {video_folder.name}")
                print(f"{'='*50}")
                
                output_dir = os.path.join(output_base_dir, video_folder.name)
                extractor.extract_silhouettes_from_folder(str(video_folder), output_dir)


if __name__ == "__main__":
    # Example usage
    frames_dir = "../data/frames"
    output_dir = "../data/silhouettes"
    
    # YOLO method (recommended for best quality)
    method = 'yolo'
    model = 'yolo11n-seg.pt'  # Options: yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg
    conf = 0.25
    device = 'cpu'  # Use 'cuda' or '0' for GPU
    
    # Alternative: Background subtraction methods
    # method = 'mog2'  # or 'knn'
    
    # Process all frame folders
    process_all_frame_folders(frames_dir, output_dir, method=method, model=model, conf=conf, device=device)
