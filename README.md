# OpenGAIT - Gait Analysis Project

## Project Structure

```
gaitlatest/
├── data/
│   ├── videos/          # Put your video files here (person-based structure)
│   │   ├── person1/
│   │   │   └── video1.mp4
│   │   ├── person2/
│   │   │   └── video2.mp4
│   │   └── person3/
│   │       └── video3.mp4
│   ├── frames/          # Extracted frames (auto-generated)
│   │   ├── person1/
│   │   │   └── video1/
│   │   │       ├── frame_000000.jpg
│   │   │       └── ...
│   │   └── person2/
│   │       └── video2/
│   └── silhouettes/     # Extracted silhouettes (auto-generated)
│       ├── person1/
│       │   └── video1/
│       └── person2/
│           └── video2/
├── scripts/
│   ├── video_to_frames.py           # Video frame extraction
│   └── silhouette_extraction.py      # Silhouette extraction
├── output/              # Analysis output
├── main.py              # Main processing pipeline
├── requirements.txt     # Python dependencies
└── venv/               # Virtual environment (to be created)
```

## Setup Instructions

### Step 1: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Required Packages

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### Step 3: Add Video Files

**OpenGait Structure (Recommended):**
```
data/videos/
├── person1/
│   └── walking_video.mp4
├── person2/
│   └── walking_video.mp4
└── person3/
    └── walking_video.mp4
```

**Alternative Flat Structure:**
```
data/videos/
├── video1.mp4
├── video2.mp4
└── video3.mp4
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.flv`, `.mkv`

### Step 4: Run Processing Pipeline

```powershell
# Run the main pipeline
python main.py
```

## What Gets Installed?

### Core Libraries
- **numpy, scipy, pandas** - Numerical and data processing
- **opencv-python** - Computer vision and video processing
- **scikit-learn, scikit-image** - Machine learning and image processing

### Deep Learning & YOLO
- **ultralytics** - YOLO object detection
- **torch, torchvision** - PyTorch deep learning framework

### Visualization & Utilities
- **matplotlib** - Plotting and visualization
- **imageio** - Image and video I/O
- **tqdm** - Progress bars

## Processing Steps

### 1. Video to Frames Extraction (`video_to_frames.py`)
- Reads video files from `data/videos/`
- Extracts frames at specified interval
- Saves frames to `data/frames/{video_name}/`

### 2. Person Detection (`detect_person.py`)
- Reads frames from `data/frames/`
- Uses YOLOv11 to detect persons
- Crops and saves detected persons
- Saves to `data/detected_persons/{video_name}/`

### 3. Silhouette Extraction (`silhouette_extraction.py`)
- Reads detected person frames from `data/detected_persons/`
- Uses background subtraction (MOG2 or KNN)
- Applies morphological operations for cleaning
- Saves silhouettes to `data/silhouettes/{video_name}/`

## Usage Examples

### Extract Frames Only
```python
from scripts.video_to_frames import extract_frames

extract_frames('data/videos/my_video.mp4', 'data/frames/my_video', frame_interval=1)
```

### Extract Silhouettes Only
```python
from scripts.silhouette_extraction import SilhouetteExtractor

extractor = SilhouetteExtractor(method='mog2')
extractor.extract_silhouettes_from_folder('data/frames/video1', 'data/silhouettes/video1')
```

### Run Full Pipeline
```powershell
python main.py
```

## Configuration

### Frame Interval
In `main.py`, modify the `frame_interval` variable:
- `1` = Extract all frames (default)
- `2` = Extract every 2nd frame
- `5` = Extract every 5th frame

### YOLO Model Selection
In `main.py`, modify the `model_name` variable:
- `'yolo11n.pt'` = Nano (fastest, default)
- `'yolo11s.pt'` = Small
- `'yolo11m.pt'` = Medium
- `'yolo11l.pt'` = Large
- `'yolo11x.pt'` = Extra Large (most accurate)

### Detection Confidence
In `main.py`, modify the `confidence` variable:
- `0.3` = Lower threshold (detect more, less accurate)
- `0.5` = Balanced (default)
- `0.7` = Higher threshold (detect less, more accurate)

### Silhouette Method
In `main.py`, modify the `method` variable:
- `'mog2'` = MOG2 background subtraction (faster, default)
- `'knn'` = KNN background subtraction (more accurate)

## Troubleshooting

### Virtual Environment Issues
```powershell
# If activation fails, try:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate again
.\venv\Scripts\Activate.ps1
```

### Package Installation Issues
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install packages one by one if needed
pip install opencv-python
pip install torch
# ... etc
```

Note about Python versions and binary wheels
------------------------------------------
If you're using Python 3.13 (or a very new Python minor version), you may encounter build errors like "Cannot import 'setuptools.build_meta'" or failures while pip attempts to build packages from source (for example `numpy==1.24.3`). This happens because many scientific packages publish prebuilt wheels only for specific Python versions. When no compatible wheel is available, pip tries to build from source which often fails on Windows.

Recommended fixes:

- Recreate the venv with a supported Python version (recommended: 3.11). Example using the Windows Python launcher if you have Python 3.11 installed:

```powershell
# remove or rename the old venv first (optional)
Remove-Item -Recurse -Force .\venv

# create a new venv with Python 3.11
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

- Or use conda/miniconda which provides prebuilt binaries for many packages:

```powershell
conda create -n opengait python=3.11
conda activate opengait
pip install -r requirements.txt
```

If you intentionally need Python 3.13, try updating `requirements.txt` to allow a newer `numpy` that provides wheels for 3.13 (if available) or install problematic packages with conda.

### CUDA Support (GPU)
If you have NVIDIA GPU and want to use CUDA:
```powershell
# Uninstall CPU versions
pip uninstall torch torchvision -y

# Install CUDA versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

After silhouette extraction:
1. ✓ Frames extracted
2. ✓ Silhouettes extracted
3. TODO: Gait feature extraction (skeleton detection)
4. TODO: Gait classification
5. TODO: Gait recognition

## Requirements File

All dependencies are listed in `requirements.txt`. To update after adding new packages:
```powershell
pip freeze > requirements.txt
```

---

**Version**: 1.0  
**Last Updated**: November 2025
