# Gait Analysis: Core Fundamentals & Concepts

## Table of Contents
1. [What is Gait?](#what-is-gait)
2. [Gait Cycle Phases](#gait-cycle-phases)
3. [Key Gait Parameters](#key-gait-parameters)
4. [Gait Features & Characteristics](#gait-features--characteristics)
5. [Silhouette Extraction Basics](#silhouette-extraction-basics)
6. [Video Processing Pipeline](#video-processing-pipeline)
7. [Machine Learning in Gait Recognition](#machine-learning-in-gait-recognition)
8. [OpenGAIT Project Structure](#opengait-project-structure)
9. [Implementation Steps](#implementation-steps)
10. [Common Terminology](#common-terminology)

---

## What is Gait?

### Definition
**Gait** is the manner/pattern of walking - the sequence of body movements that produce forward locomotion.

### Why is Gait Recognition Important?
- **Biometric Identification**: Like fingerprints, each person has a unique gait pattern
- **Security Applications**: Surveillance, access control, criminal identification
- **Medical Applications**: Detecting neurological disorders, rehabilitation monitoring
- **Behavioral Analysis**: Understanding movement patterns, health assessment

### Key Insight
> "Gait is one of the few biometrics that can be recognized from a distance without the subject's knowledge or cooperation."

---

## Gait Cycle Phases

### The Complete Gait Cycle (100% of one step = 1 gait cycle)

```
STANCE PHASE (60% of cycle)          SWING PHASE (40% of cycle)
└─ Single Support (60%)              └─ Single Support (40%)
   ├─ Initial Contact (0-2%)            ├─ Initial Swing (60-73%)
   ├─ Loading Response (0-10%)          ├─ Mid Swing (73-87%)
   ├─ Mid Stance (10-30%)               └─ Terminal Swing (87-100%)
   ├─ Terminal Stance (30-50%)
   └─ Pre-swing (50-60%)
```

### Phase Details

#### 1. **Initial Contact (0-2%)**
- One foot touches the ground
- Beginning of stance phase
- Heel strike on the ground
- Other leg is swinging forward

#### 2. **Loading Response (0-10%)**
- Weight is transferred to the newly planted foot
- Knee and ankle absorb shock
- Both feet briefly in contact with ground

#### 3. **Mid Stance (10-30%)**
- Body weight is directly over the supporting leg
- Pelvis is level and stable
- Swing leg moving forward

#### 4. **Terminal Stance (30-50%)**
- Body continues forward over the support foot
- Heel of supporting foot may lift off
- Opposite leg preparing to land

#### 5. **Pre-swing (50-60%)**
- Weight begins transferring to the other leg
- Push-off phase with toes
- Preparing for swing phase

#### 6. **Initial Swing (60-73%)**
- Foot leaves the ground after push-off
- Hip and knee flexion begins
- Acceleration phase

#### 7. **Mid Swing (73-87%)**
- Swinging leg passes under the body
- Hip continues to flex
- Maximum knee flexion occurs

#### 8. **Terminal Swing (87-100%)**
- Swinging leg extends in preparation for ground contact
- Ready for initial contact with ground
- Knee extends

---

## Key Gait Parameters

### Distance Parameters
| Parameter | Definition | Normal Value |
|-----------|-----------|--------------|
| **Stride Length** | Distance from one heel strike to the next same foot | 1.4-1.6 m |
| **Step Length** | Distance from one foot to the other | 0.7-0.8 m |
| **Step Width** | Lateral distance between feet | 0.08-0.15 m |
| **Stride Width** | Perpendicular distance between the lines of walking | 0.1-0.2 m |

### Time Parameters
| Parameter | Definition | Normal Value |
|-----------|-----------|--------------|
| **Cadence** | Number of steps per minute | 100-130 steps/min |
| **Gait Cycle Time** | Time to complete one full cycle | 0.9-1.1 seconds |
| **Stance Time** | Time foot is in contact with ground | 0.5-0.7 seconds |
| **Swing Time** | Time foot is off the ground | 0.4-0.5 seconds |

### Velocity Parameters
| Parameter | Definition | Normal Value |
|-----------|-----------|--------------|
| **Walking Speed** | Distance traveled per unit time | 1.2-1.5 m/s |
| **Step Frequency** | Steps per second | 1.67-2.17 steps/s |

---

## Gait Features & Characteristics

### 1. **Spatio-Temporal Features**
Features that describe movement across space and time:
- **Joint angles**: Hip, knee, ankle flexion/extension
- **Joint velocities**: Rate of change of joint angles
- **Segment angles**: Positions of body segments (leg, thigh, torso)
- **Center of mass trajectory**: Path of body's center of gravity

### 2. **Silhouette-Based Features**
Extracted from silhouette (body outline):
- **Silhouette centroid**: Center of mass position
- **Silhouette width**: Distance between left and right edges
- **Silhouette height**: Distance from top to bottom
- **Silhouette area**: Total pixels in silhouette
- **Silhouette boundary**: Perimeter and shape descriptors

### 3. **Skeleton-Based Features**
Extracted from joint positions (requires pose detection):
- **Joint coordinates**: Position of each joint (hip, knee, ankle, etc.)
- **Limb angles**: Angle between connected joints
- **Limb lengths**: Distance between joints
- **Joint distances**: Euclidean distances between joints

### 4. **Energy & Motion Features**
- **Optical flow**: Movement vectors in frames
- **Motion history**: Temporal accumulation of motion
- **Energy expenditure**: Estimated based on movement

### 5. **Distinctive Characteristics**
- **Stride length variability**
- **Gait asymmetry** (differences between left/right)
- **Heel strike pattern**
- **Toe-off pattern**
- **Arm swing amplitude**
- **Torso lean**
- **Head bob** (vertical oscillation)

---

## Silhouette Extraction Basics

### What is a Silhouette?
A **silhouette** is a binary (black-and-white) image of a person's outline/shape, obtained by:
1. Subtracting the background from the current frame
2. Creating a mask where the person is white (255) and background is black (0)

### Why Extract Silhouettes?
- **Remove appearance variations**: Clothing, lighting, color don't matter
- **Focus on body shape and motion**: Extract gait patterns
- **Reduce computational complexity**: Binary image is smaller and faster to process
- **Robustness**: Works in uncontrolled environments

### Common Background Subtraction Methods

#### 1. **MOG2 (Mixture of Gaussians v2)**
```python
cv2.createBackgroundSubtractorMOG2(
    detectShadows=True,      # Remove shadow pixels
    varThreshold=16,         # Variance threshold (default: 16)
    history=500              # Number of frames to learn background
)
```
**Pros**: Fast, good for real-time, handles gradual changes  
**Cons**: Struggles with sudden lighting changes, shadows

#### 2. **KNN (K-Nearest Neighbors)**
```python
cv2.createBackgroundSubtractorKNN(
    detectShadows=True,      # Remove shadow pixels
    history=500              # Number of frames
)
```
**Pros**: Adaptive, good shadow detection, handles moving backgrounds  
**Cons**: Slower than MOG2, needs more memory

### Morphological Operations for Silhouette Cleaning

After background subtraction, apply morphological operations:

#### 1. **Opening** (Erosion → Dilation)
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
```
**Effect**: Removes small noise/holes outside the silhouette

#### 2. **Closing** (Dilation → Erosion)
```python
filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
```
**Effect**: Fills small holes inside the silhouette

#### 3. **Other Operations**
- **Erosion**: Shrinks white regions
- **Dilation**: Expands white regions
- **Gradient**: Finds boundaries
- **Tophat**: Detects small objects
- **Blackhat**: Detects dark objects

---

## Video Processing Pipeline

### Step 1: Video Input
```
Input Video File (MP4, AVI, MOV, etc.)
    ↓
Read video properties (FPS, frame count, resolution)
    ↓
Extract frames or process frame-by-frame
```

### Step 2: Frame Extraction
```
Video Stream
    ↓
For each frame:
├─ Read frame
├─ Resize (optional, for speed)
├─ Convert to appropriate color space (RGB, BGR, Grayscale)
└─ Save as image file
```

### Step 3: Silhouette Extraction
```
Extracted Frames
    ↓
For each frame:
├─ Apply background subtraction → Binary mask
├─ Apply morphological operations → Cleaned silhouette
├─ Apply edge detection (optional) → Boundary extraction
└─ Save silhouette image
```

### Step 4: Feature Extraction
```
Silhouettes
    ↓
For each silhouette:
├─ Extract spatio-temporal features
├─ Extract shape features
├─ Extract boundary features
└─ Store in feature vector
```

### Step 5: Gait Recognition/Classification
```
Feature Vectors
    ↓
Machine Learning Model
├─ Training: Learn from known individuals
├─ Testing: Identify or verify individuals
└─ Output: Recognition result or classification
```

---

## Machine Learning in Gait Recognition

### Approaches

#### 1. **Traditional Machine Learning**
Use hand-crafted features + classical models:

```python
# Feature Extraction
features = extract_gait_features(silhouettes)

# Training
model = RandomForest()  # or SVM, KNN, etc.
model.fit(X_train, y_train)

# Recognition
prediction = model.predict(features)
```

**Models**: Random Forest, SVM, KNN, Logistic Regression, Decision Trees

#### 2. **Deep Learning (Recommended for modern systems)**
Learn features automatically:

```python
# CNN for feature learning
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(height, width, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_people, activation='softmax')
])

# Training
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=50)

# Recognition
prediction = model.predict(test_silhouettes)
```

**Models**: CNN, RNN, LSTM (for temporal sequences), Siamese Networks, 3D CNN

#### 3. **Person Re-identification (ReID) Approaches**
View the gait recognition as a person re-identification problem:
- Use metric learning (triplet loss, center loss)
- Learn discriminative embeddings
- Match based on similarity in embedding space

### Gait Recognition Tasks

#### 1. **Identification** (Closed-set)
"Who is this person?"
- Match silhouette sequence against all known individuals
- Output: Person ID with confidence score
- Use-case: Surveillance, known criminal database

#### 2. **Verification** (1:1 matching)
"Is this the same person?"
- Compare two gait sequences
- Output: Same/Different with confidence
- Use-case: Access control, authentication

#### 3. **Retrieval** (Ranking)
"Show me similar walkers"
- Rank all individuals by similarity to query gait
- Output: Ranked list of candidates
- Use-case: Finding suspicious individuals

---

## OpenGAIT Project Structure

### Folder Organization
```
gaitlatest/
├── data/
│   ├── videos/          # Raw video files
│   │   ├── person1/
│   │   ├── person2/
│   │   └── person3/
│   ├── frames/          # Extracted frames (auto-generated)
│   │   ├── person1/
│   │   │   └── video1/
│   │   │       ├── frame_000000.jpg
│   │   │       └── ...
│   │   └── person2/
│   └── silhouettes/     # Extracted silhouettes (auto-generated)
│       ├── person1/
│       │   └── video1/
│       │       ├── frame_000000_silhouette.jpg
│       │       └── ...
│       └── person2/
├── scripts/
│   ├── video_to_frames.py          # Extract frames from video
│   └── silhouette_extraction.py    # Extract silhouettes from frames
├── output/              # Analysis results
├── main.py              # Main processing pipeline
├── requirements.txt     # Python dependencies
└── venv/               # Virtual environment
```

### Key Python Libraries

```
numpy              # Numerical operations, array handling
scipy              # Scientific computing (statistics, optimization)
pandas             # Data manipulation and analysis
opencv-python     # Computer vision (video I/O, image processing, background subtraction)
scikit-learn      # Machine learning (classification, clustering)
scikit-image      # Image processing (morphological ops, features)
torch, torchvision # Deep learning (neural networks, CNN models)
ultralytics       # YOLO object detection
matplotlib        # Visualization and plotting
tqdm              # Progress bars
imageio           # Image and video I/O
```

---

## Implementation Steps

### Phase 1: Setup & Data Preparation (Week 1)
- [x] Create project structure
- [x] Create virtual environment
- [ ] Install required packages (fix Python version issue first)
- [ ] Collect or prepare video data
- [ ] Organize videos in `data/videos/` folder

### Phase 2: Video Processing (Week 2-3)
- [ ] Implement frame extraction from videos
- [ ] Save extracted frames to `data/frames/`
- [ ] Implement video frame interval configuration
- [ ] Test on sample videos

### Phase 3: Silhouette Extraction (Week 3-4)
- [ ] Implement background subtraction (MOG2, KNN)
- [ ] Implement morphological operations (opening, closing)
- [ ] Extract silhouettes from frames
- [ ] Save silhouettes to `data/silhouettes/`
- [ ] Compare different background subtraction methods

### Phase 4: Feature Extraction (Week 4-5)
- [ ] Extract spatio-temporal features from silhouettes
- [ ] Extract shape-based features (centroid, width, height, etc.)
- [ ] Implement feature normalization
- [ ] Create feature vectors for each video sequence

### Phase 5: Model Training (Week 5-6)
- [ ] Split data into train/test sets
- [ ] Train machine learning classifier
- [ ] Evaluate model performance
- [ ] Implement gait recognition/verification

### Phase 6: Recognition & Testing (Week 6-7)
- [ ] Implement person identification
- [ ] Implement person verification
- [ ] Test on new video sequences
- [ ] Generate recognition results and visualizations

---

## Common Terminology

### Basic Terms
| Term | Definition |
|------|-----------|
| **Gait** | Manner of walking; sequence of body movements |
| **Silhouette** | Binary outline/shape of person extracted from frame |
| **Pose** | Spatial configuration of body joints and limbs |
| **Skeleton** | Set of joint positions representing body structure |
| **Motion Capture** | Recording detailed movement of body/objects |
| **Biometric** | Measurable biological characteristic for identification |

### Image Processing Terms
| Term | Definition |
|------|-----------|
| **Background Subtraction** | Technique to separate foreground (person) from background |
| **Foreground** | Moving objects of interest (person walking) |
| **Background** | Static scene (walls, floor, furniture) |
| **Binary Image** | Image with only 2 values (0=black, 255=white) |
| **Morphological Ops** | Operations that process images based on shapes |
| **Erosion** | Shrinking white regions in binary image |
| **Dilation** | Expanding white regions in binary image |
| **Opening** | Erosion followed by dilation; removes small noise |
| **Closing** | Dilation followed by erosion; fills small holes |

### Machine Learning Terms
| Term | Definition |
|------|-----------|
| **Classification** | Predicting category/label for input data |
| **Feature** | Individual measurable property used for analysis |
| **Training Set** | Data used to train the model |
| **Test Set** | Data used to evaluate model performance |
| **Accuracy** | Percentage of correct predictions |
| **Precision** | Of positive predictions, how many were correct |
| **Recall** | Of actual positives, how many were detected |
| **F1-Score** | Harmonic mean of precision and recall |

### Gait Recognition Terms
| Term | Definition |
|------|-----------|
| **Gait Cycle** | Complete walking pattern from one heel strike to the next (same foot) |
| **Stride** | Distance or time for one complete gait cycle |
| **Step** | Distance or time from one foot strike to the other foot strike |
| **Cadence** | Number of steps per unit time (usually per minute) |
| **Spatio-Temporal** | Relating to both space (position) and time (movement over frames) |
| **Covariate** | Condition that changes (viewing angle, clothing, carrying items, etc.) |
| **In-the-wild** | Real-world unconstrained conditions (vs controlled lab) |

---

## Key Concepts to Master

### 1. **Gait Cycle Understanding**
- Memorize the phases and percentages
- Understand what's happening at each phase
- Know typical duration and patterns

### 2. **Silhouette Extraction**
- How background subtraction works
- When to use MOG2 vs KNN
- Why morphological operations are needed
- How to clean and prepare silhouettes

### 3. **Feature Extraction**
- What makes a good gait feature
- Difference between handcrafted vs learned features
- How to normalize and prepare features

### 4. **Machine Learning in Gait**
- Classification vs identification vs verification
- Evaluation metrics for gait recognition
- How to handle biometric data and privacy

### 5. **Implementation**
- Complete pipeline from video to recognition
- Proper data organization
- Debugging and validation

---

## Study Tips

1. **Visualize the Process**
   - Watch videos of people walking
   - Look at extracted frames
   - Examine silhouettes
   - See the differences between individuals

2. **Experiment with Parameters**
   - Try different frame intervals
   - Test different background subtraction methods
   - Adjust morphological operation sizes
   - See how parameters affect output

3. **Understand the Math**
   - Review statistics and linear algebra basics
   - Understand distance metrics (Euclidean, Cosine, etc.)
   - Learn about optimization for model training

4. **Read Research Papers**
   - OpenGAIT original papers on arXiv
   - CASIA-B gait dataset papers
   - Recent gait recognition benchmarks

5. **Practice Implementation**
   - Modify scripts with different parameters
   - Combine multiple features
   - Test on different videos
   - Compare results across methods

---

## Quick Reference Checklist

- [ ] Understand all 8 phases of gait cycle
- [ ] Know normal values for gait parameters
- [ ] Explain why silhouettes are useful
- [ ] Describe MOG2 and KNN background subtraction
- [ ] Know what morphological operations do
- [ ] Understand spatio-temporal vs static features
- [ ] Explain classification vs identification vs verification
- [ ] Describe complete processing pipeline
- [ ] Set up and run video frame extraction
- [ ] Set up and run silhouette extraction
- [ ] Extract features from silhouettes
- [ ] Train a simple gait classifier
- [ ] Test gait recognition on new videos

---

## Resources & Next Steps

### Online Resources
- OpenGAIT GitHub: https://github.com/ShiqiYu/OpenGAIT
- CASIA-B Dataset: http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp
- OpenCV Background Subtraction: https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html

### Next Session Topics
1. Setting up proper Python 3.11 environment
2. Installing all required packages successfully
3. Processing first video file
4. Extracting and visualizing frames
5. Extracting silhouettes with different methods
6. Comparing silhouette quality

### Questions to Think About
- Why is gait unique to each person?
- What factors affect gait patterns?
- Why might gait recognition fail in certain scenarios?
- How would you handle people walking at different speeds?
- How would you recognize the same person from different viewing angles?

---

**Last Updated**: November 11, 2025  
**Version**: 1.0  
**Status**: Ready for Study & Reference
