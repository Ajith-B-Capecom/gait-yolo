"""
DeepSORT Tracking Handler
Separate module for DeepSORT-specific tracking operations
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pkg_resources')

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False


class DeepSortHandler:
    """Handles DeepSORT tracking operations"""
    
    def __init__(self):
        """Initialize DeepSORT tracker with optimized parameters"""
        if not DEEPSORT_AVAILABLE:
            raise ImportError(
                "deep_sort_realtime not installed. "
                "Install with: pip install deep-sort-realtime"
            )
        
        print("Initializing DeepSORT tracker...")
        self.tracker = DeepSort(
            max_age=20,              # Max frames to keep lost tracks
            n_init=2,                # Min detections before track confirmation
            nms_max_overlap=0.3,     # NMS threshold
            max_cosine_distance=0.8, # Appearance matching threshold
            nn_budget=None,          # Feature vector budget
            override_track_class=None,
            embedder="mobilenet",    # Feature extractor
            half=True,               # Use FP16 for speed
            bgr=True,                # OpenCV uses BGR
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        print("DeepSORT tracker initialized successfully!")
    
    @staticmethod
    def convert_yolo_to_deepsort_format(boxes):
        """
        Convert YOLO boxes to DeepSORT format
        
        Args:
            boxes: YOLO detection boxes
            
        Returns:
            List of detections in format: ([x, y, w, h], confidence, class)
        """
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # DeepSORT expects: ([left, top, width, height], confidence, class)
            detections.append(([x1, y1, w, h], conf, cls))
        
        return detections
    
    def update_tracks(self, detections, frame):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections in DeepSORT format
            frame: Current video frame
            
        Returns:
            track_ids: List of tracking IDs
            track_boxes: List of bounding boxes [x1, y1, x2, y2]
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        track_ids = []
        track_boxes = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_ids.append(track.track_id)
            track_boxes.append(track.to_ltrb())  # [x1, y1, x2, y2]
        
        return track_ids, track_boxes
    
    @staticmethod
    def match_keypoints_to_tracks(keypoints, yolo_boxes, track_boxes, track_ids, iou_threshold=0.3):
        """
        Match YOLO keypoints to DeepSORT tracking IDs based on bbox IoU
        
        Args:
            keypoints: YOLO pose keypoints array
            yolo_boxes: YOLO detection boxes
            track_boxes: DeepSORT tracking boxes
            track_ids: DeepSORT tracking IDs
            iou_threshold: Minimum IoU for matching
            
        Returns:
            List of matched data with keypoints and track IDs
        """
        matched_data = []
        
        for kpt_idx, person_kpts in enumerate(keypoints):
            if kpt_idx >= len(yolo_boxes):
                break
                
            yolo_box = yolo_boxes[kpt_idx]
            yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_box.xyxy[0].cpu().numpy()
            
            best_iou = 0
            best_track_id = None
            
            # Find best matching track based on IoU
            for track_box, track_id in zip(track_boxes, track_ids):
                track_x1, track_y1, track_x2, track_y2 = track_box
                
                # Calculate IoU
                x1 = max(yolo_x1, track_x1)
                y1 = max(yolo_y1, track_y1)
                x2 = min(yolo_x2, track_x2)
                y2 = min(yolo_y2, track_y2)
                
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)
                track_area = (track_x2 - track_x1) * (track_y2 - track_y1)
                union = yolo_area + track_area - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            # Only match if IoU is above threshold
            if best_iou > iou_threshold and best_track_id is not None:
                matched_data.append({
                    'keypoints': person_kpts,
                    'track_id': int(best_track_id)  # Ensure it's an integer
                })
        
        return matched_data
    
    def process_frame_with_deepsort(self, frame, pose_model, conf_threshold=0.5):
        """
        Complete DeepSORT processing pipeline for a single frame
        
        Args:
            frame: Video frame
            pose_model: YOLO pose model
            conf_threshold: Confidence threshold
            
        Returns:
            matched_data: List of dicts with keypoints and track_ids
        """
        # Detect poses with YOLO
        results = pose_model(frame, conf=conf_threshold, verbose=False)
        
        # Get detections for DeepSORT
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Convert to DeepSORT format
            detections = self.convert_yolo_to_deepsort_format(results[0].boxes)
            
            # Update tracker
            track_ids, track_boxes = self.update_tracks(detections, frame)
            
            # Match keypoints to tracks
            if results[0].keypoints is not None:
                keypoints = results[0].keypoints.data.cpu().numpy()
                matched_data = self.match_keypoints_to_tracks(
                    keypoints, results[0].boxes, track_boxes, track_ids
                )
            else:
                matched_data = []
        else:
            matched_data = []
        
        return matched_data


def is_deepsort_available():
    """Check if DeepSORT is available"""
    return DEEPSORT_AVAILABLE


def create_deepsort_handler():
    """Factory function to create DeepSORT handler"""
    if not DEEPSORT_AVAILABLE:
        raise ImportError(
            "DeepSORT is not available. Install with: pip install deep-sort-realtime"
        )
    return DeepSortHandler()