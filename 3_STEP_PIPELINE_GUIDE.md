# 3-Step OpenGait Pipeline Guide

## 📋 Overview

Complete pipeline with 3 distinct steps:
1. **Video → Frames** - Extract frames from videos
2. **Frames → Person Detection** - Detect and crop persons using YOLOv11
3. **Person Crops → Silhouettes** - Extract silhouettes from cropped persons

---

## 🔄 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Video to Frames                                        │
└─────────────────────────────────────────────────────────────────┘
   Input:  data/videos/person1/walking.mp4
           ↓
   Output: data/frames/person1/walking/frame_000000.jpg
                                        frame_000001.jpg
                                        ...

┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Person Detection (YOLOv11)                             │
└─────────────────────────────────────────────────────────────────┘
   Input:  data/frames/person1/walking/frame_000000.jpg
           ↓
   Output: data/detected_persons/person1/walking/
           ├── bbox_visualizations/     (frames with boxes drawn)
           │   └── frame_000000.jpg
           └── person_crops/            (cropped person images)
               ├── frame_000000_person0.jpg
               └── frame_000001_person0.jpg

┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Silhouette Extraction                                  │
└─────────────────────────────────────────────────────────────────┘
   Input:  data/detected_persons/person1/walking/person_crops/
           ↓
   Output: data/silhouettes/person1/walking/
           └── frame_000000_person0_silhouette.jpg
               frame_000001_person0_silhouette.jpg
               ...
```

---

## 🚀 Quick Start

### Run Complete Pipeline

```powershell
python main.py
```

This will automatically run all 3 steps.

---

## ⚙️ Configuration

### In main.py

```python
# STEP 1: Frame Extraction (line ~80)
frame_interval = 1  # Extract every frame (1=all, 2=every 2nd, 5=every 5th)

# STEP 2: Person Detection (line ~95)
detection_model = 'yolo11n.pt'  # Model: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
detection_conf = 0.5            # Confidence threshold (0.1-0.9)
device = 'cpu'                  # Device: 'cpu', 'cuda', '0', '1'
save_bbox_images = True         # Save visualization with boxes
save_crops = True               # Save cropped person images

# STEP 3: Silhouette Extraction (line ~120)
method = 'yolo'                 # Method: 'yolo', 'mog2', 'knn'
silhouette_model = 'yolo11n-seg.pt'  # YOLO segmentation model
silhouette_conf = 0.25          # Confidence threshold
```

---

## 📊 Output Structure

After running the complete pipeline:

```
data/
├── videos/
│   └── person1/
│       └── walking.mp4                          [INPUT]
│
├── frames/
│   └── person1/
│       └── walking/
│           ├── frame_000000.jpg                 [STEP 1 OUTPUT]
│           ├── frame_000001.jpg
│           └── ...
│
├── detected_persons/
│   └── person1/
│       └── walking/
│           ├── bbox_visualizations/             [STEP 2 OUTPUT]
│           │   ├── frame_000000.jpg             (with boxes drawn)
│           │   └── ...
│           └── person_crops/                    [STEP 2 OUTPUT]
│               ├── frame_000000_person0.jpg     (cropped person)
│               ├── frame_000001_person0.jpg
│               └── ...
│
└── silhouettes/
    └── person1/
        └── walking/
            ├── frame_000000_person0_silhouette.jpg  [STEP 3 OUTPUT]
            ├── frame_000001_person0_silhouette.jpg
            └── ...
```

---

## 🎯 Step-by-Step Details

### STEP 1: Video to Frames

**Script:** `scripts/video_to_frames.py`

**What it does:**
- Reads video files
- Extracts frames at specified interval
- Saves as JPG images

**Configuration:**
```python
frame_interval = 1  # All frames
frame_interval = 2  # Every 2nd frame (half speed)
frame_interval = 5  # Every 5th frame (1/5 speed)
```

**Run individually:**
```powershell
python scripts/video_to_frames.py
```

---

### STEP 2: Person Detection

**Script:** `scripts/detect_person.py`

**What it does:**
- Detects persons in each frame using YOLOv11
- Draws bounding boxes (optional)
- Crops detected persons (optional)
- Handles multiple persons per frame

**Models:**
- `yolo11n.pt` - Nano (fastest, 2.5 MB)
- `yolo11s.pt` - Small (balanced, 9.4 MB)
- `yolo11m.pt` - Medium (good quality, 20 MB)
- `yolo11l.pt` - Large (high quality, 26 MB)
- `yolo11x.pt` - Extra large (best, 30 MB)

**Configuration:**
```python
detection_model = 'yolo11n.pt'  # Choose model
detection_conf = 0.5            # Higher = fewer detections
save_bbox_images = True         # Save visualizations
save_crops = True               # Save cropped persons
```

**Run individually:**
```powershell
python scripts/detect_person.py
```

**Output:**
- `bbox_visualizations/` - Frames with green boxes around persons
- `person_crops/` - Individual cropped person images

---

### STEP 3: Silhouette Extraction

**Script:** `scripts/silhouette_extraction.py`

**What it does:**
- Extracts silhouettes from cropped person images
- Uses YOLO segmentation or background subtraction
- Creates binary masks (white person on black background)

**Methods:**

**Option 1: YOLO Segmentation (Recommended)**
```python
method = 'yolo'
silhouette_model = 'yolo11n-seg.pt'  # Segmentation model
silhouette_conf = 0.25
```

**Models:**
- `yolo11n-seg.pt` - Nano (fastest)
- `yolo11s-seg.pt` - Small
- `yolo11m-seg.pt` - Medium
- `yolo11l-seg.pt` - Large
- `yolo11x-seg.pt` - Extra large

**Option 2: Background Subtraction (Faster)**
```python
method = 'mog2'  # or 'knn'
```

**Run individually:**
```powershell
python scripts/silhouette_extraction.py
```

---

## 🎨 Visual Examples

### STEP 1 Output: Frames
```
Original video frame extracted as image
[Full scene with person and background]
```

### STEP 2 Output: Detection

**Bounding Box Visualization:**
```
┌─────────────────────────┐
│                         │
│    ┌──────────┐         │
│    │ Person   │         │  ← Green box around person
│    │  0.95    │         │  ← Confidence score
│    └──────────┘         │
│                         │
└─────────────────────────┘
```

**Cropped Person:**
```
┌──────────┐
│          │
│  Person  │  ← Just the person, cropped
│          │
└──────────┘
```

### STEP 3 Output: Silhouette
```
┌──────────┐
│  ▓▓▓▓▓▓  │
│  ▓▓▓▓▓▓  │  ← White silhouette
│  ▓▓▓▓▓▓  │  ← Black background
│  ▓▓▓▓▓▓  │
└──────────┘
```

---

## 💡 Why 3 Steps?

### Benefits of Separate Detection Step

1. **Better Silhouettes**
   - Silhouette extraction works on cropped persons
   - No background interference
   - Cleaner results

2. **Multiple Persons**
   - Each person detected separately
   - Individual crops for each person
   - Separate silhouettes per person

3. **Debugging**
   - Can inspect detection quality
   - Visualize bounding boxes
   - Verify crops before silhouette extraction

4. **Flexibility**
   - Can skip detection if not needed
   - Can use different models for detection vs segmentation
   - Can reprocess silhouettes without re-detecting

---

## 🔧 Model Selection Guide

### For Detection (STEP 2)

**CPU:**
```python
detection_model = 'yolo11n.pt'  # Fastest
detection_conf = 0.5
```

**GPU:**
```python
detection_model = 'yolo11m.pt'  # Balanced
detection_conf = 0.5
device = 'cuda'
```

### For Silhouettes (STEP 3)

**Best Quality:**
```python
method = 'yolo'
silhouette_model = 'yolo11m-seg.pt'
device = 'cuda'
```

**Best Speed:**
```python
method = 'mog2'  # Background subtraction
```

**Balanced:**
```python
method = 'yolo'
silhouette_model = 'yolo11n-seg.pt'
device = 'cpu'
```

---

## 📈 Performance

### Processing Time (300 frames, 1920x1080)

**STEP 1: Frame Extraction**
- Time: ~5 seconds
- Output: 300 frames

**STEP 2: Person Detection**
- CPU (yolo11n): ~30 seconds
- GPU (yolo11n): ~5 seconds
- Output: 300 detections + crops

**STEP 3: Silhouette Extraction**
- YOLO (CPU): ~45 seconds
- YOLO (GPU): ~8 seconds
- MOG2: ~5 seconds
- Output: 300 silhouettes

**Total Time:**
- CPU: ~80 seconds
- GPU: ~18 seconds

---

## 🐛 Troubleshooting

### No Persons Detected

**Problem:** STEP 2 finds no persons

**Solutions:**
1. Lower confidence threshold:
   ```python
   detection_conf = 0.3  # or 0.2
   ```

2. Check bbox_visualizations to see if boxes are drawn

3. Verify frames are not corrupted

### Poor Silhouette Quality

**Problem:** STEP 3 produces noisy silhouettes

**Solutions:**
1. Use YOLO segmentation instead of background subtraction:
   ```python
   method = 'yolo'
   ```

2. Lower confidence threshold:
   ```python
   silhouette_conf = 0.1
   ```

3. Use larger model:
   ```python
   silhouette_model = 'yolo11m-seg.pt'
   ```

### Multiple Persons in Frame

**Problem:** Multiple persons detected, need to track specific person

**Solution:**
- Detection automatically creates separate crops for each person
- Each crop gets its own silhouette
- Files named: `frame_000000_person0.jpg`, `frame_000000_person1.jpg`, etc.

### Slow Processing

**Solutions:**
1. Increase frame interval:
   ```python
   frame_interval = 5  # Process fewer frames
   ```

2. Use smaller models:
   ```python
   detection_model = 'yolo11n.pt'
   silhouette_model = 'yolo11n-seg.pt'
   ```

3. Enable GPU:
   ```python
   device = 'cuda'
   ```

4. Use background subtraction for silhouettes:
   ```python
   method = 'mog2'
   ```

---

## 🎯 Use Cases

### Research / High Quality Dataset
```python
# STEP 1
frame_interval = 1

# STEP 2
detection_model = 'yolo11l.pt'
detection_conf = 0.5
device = 'cuda'

# STEP 3
method = 'yolo'
silhouette_model = 'yolo11l-seg.pt'
device = 'cuda'
```

### Quick Testing / Prototyping
```python
# STEP 1
frame_interval = 10  # Every 10th frame

# STEP 2
detection_model = 'yolo11n.pt'
detection_conf = 0.5
device = 'cpu'

# STEP 3
method = 'mog2'  # Fast background subtraction
```

### Production / Real-time
```python
# STEP 1
frame_interval = 2

# STEP 2
detection_model = 'yolo11s.pt'
detection_conf = 0.6
device = 'cuda'

# STEP 3
method = 'yolo'
silhouette_model = 'yolo11n-seg.pt'
device = 'cuda'
```

---

## 📚 Related Documentation

- **YOLO_SILHOUETTE_GUIDE.md** - Detailed silhouette extraction guide
- **README.md** - Complete project documentation
- **QUICKSTART.md** - Quick reference
- **WORKFLOW.md** - Step-by-step workflow

---

## ✅ Checklist

- [ ] Videos added to `data/videos/person1/`, `person2/`, etc.
- [ ] Configuration updated in `main.py`
- [ ] Run: `python main.py`
- [ ] Check frames in `data/frames/`
- [ ] Check detections in `data/detected_persons/bbox_visualizations/`
- [ ] Check crops in `data/detected_persons/person_crops/`
- [ ] Check silhouettes in `data/silhouettes/`
- [ ] Adjust confidence thresholds if needed
- [ ] Enable GPU for faster processing (optional)

---

**Ready to process gait videos with the complete 3-step pipeline!** 🎉
