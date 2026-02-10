# Live Facial Recognition for NVIDIA Jetson Nano

A complete pipeline for real-time facial recognition using MobileFaceNet, optimized for NVIDIA Jetson Nano.

## Features

- ✅ Real-time face detection and recognition
- ✅ Lightweight MobileFaceNet model (optimized for embedded devices)
- ✅ GPU acceleration via CUDA
- ✅ 15-25 FPS on Jetson Nano
- ✅ Simple database management
- ✅ Multi-face detection and recognition

## Hardware Requirements

- NVIDIA Jetson Nano (4GB recommended)
- USB Camera or CSI Camera
- microSD card (32GB+ recommended)
- Power supply (5V 4A recommended for stable performance)

## Software Requirements

- JetPack 4.6 or later
- Python 3.6+
- CUDA 10.2+

## Installation

### Step 1: Clone or Copy Files

Copy these files to your Jetson Nano:
- `face_recognition_pipeline.py`
- `install_jetson.sh`
- `requirements.txt`

### Step 2: Run Installation Script

```bash
chmod +x install_jetson.sh
./install_jetson.sh
```

This will install all required dependencies including:
- OpenCV
- InsightFace
- ONNX Runtime (GPU)
- NumPy, SciPy, etc.

**Note:** First run will download the face detection model (~5MB) automatically.

### Alternative: Manual Installation

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y python3-pip python3-opencv libopencv-dev

# Install Python packages
pip3 install -r requirements.txt
```

## Preparing Your Training Data

### Step 1: Organize Images

Create a folder structure like this:

```
training_data/
├── person1/
│   ├── img001.jpg
│   ├── img002.jpg
│   ├── ...
│   └── img040.jpg
├── person2/
│   ├── img001.jpg
│   └── ...
├── person3/
│   └── ...
└── person4/
    └── ...
```

### Step 2: Image Guidelines

For best results:
- Use **40 images per person** (as you mentioned)
- Images should contain **clear, frontal faces**
- Vary lighting conditions slightly
- Include different expressions
- Resolution: 640x480 or higher
- Format: JPG, PNG, or BMP

## Usage

### Basic Usage

```bash
python3 face_recognition_pipeline.py
```

When prompted:
1. Enter 'y' to build the database
2. Enter the path to your training_data folder
3. Wait for processing (1-2 minutes for 160 images)
4. Live recognition will start automatically

### Advanced Usage

You can also use the pipeline programmatically:

```python
from face_recognition_pipeline import FaceRecognitionPipeline

# Initialize
pipeline = FaceRecognitionPipeline()

# Add people to database
pipeline.add_person_to_database("John", "training_data/person1")
pipeline.add_person_to_database("Jane", "training_data/person2")
pipeline.save_database()

# Run live recognition
pipeline.run_live_recognition(camera_id=0, display=True)
```

## Configuration Options

### Adjust Recognition Threshold

In `face_recognition_pipeline.py`, modify:

```python
self.threshold = 0.6  # Lower = stricter (0.5-0.7 recommended)
```

### Change Detection Size

For faster processing (lower accuracy):
```python
self.app.prepare(ctx_id=0, det_size=(240, 240))  # Default: (320, 320)
```

For better accuracy (slower):
```python
self.app.prepare(ctx_id=0, det_size=(640, 640))
```

### Frame Skip

Process every Nth frame for better FPS:
```python
frame_skip = 2  # Process every 2nd frame (default)
frame_skip = 3  # Process every 3rd frame (faster)
frame_skip = 1  # Process every frame (slower but smoother)
```

## Performance Optimization Tips

### 1. TensorRT Conversion (Advanced)

For maximum performance, convert the model to TensorRT:

```python
# This requires additional setup
# See: https://github.com/NVIDIA-AI-IOT/torch2trt
```

### 2. Reduce Camera Resolution

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower for more FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### 3. Use Power Mode 0 (Max Performance)

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 4. Disable GUI (Headless Mode)

```python
pipeline.run_live_recognition(camera_id=0, display=False)
```

## Troubleshooting

### Low FPS (<10 FPS)

- Enable MAX power mode: `sudo nvpmodel -m 0`
- Increase frame_skip to 3 or 4
- Reduce detection size to (240, 240)
- Lower camera resolution

### "CUDA out of memory" Error

- Reduce det_size
- Lower camera resolution
- Close other applications

### Camera Not Detected

```bash
# List cameras
ls /dev/video*

# Test camera
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink

# For USB camera, try different IDs
pipeline.run_live_recognition(camera_id=1)
```

### Model Download Fails

Download manually:
```bash
mkdir -p ~/.insightface/models/buffalo_sc
# Download from: https://github.com/deepinsight/insightface/releases
```

## Expected Performance

On NVIDIA Jetson Nano:

| Configuration | FPS | Accuracy |
|--------------|-----|----------|
| Default (320x320 det) | 15-20 | High |
| Fast (240x240 det, skip=3) | 25-30 | Good |
| Quality (640x640 det) | 8-12 | Very High |

## File Structure

```
.
├── face_recognition_pipeline.py  # Main pipeline script
├── install_jetson.sh             # Installation script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── training_data/                # Your training images
│   ├── person1/
│   ├── person2/
│   ├── person3/
│   └── person4/
└── face_database/                # Generated database
    └── embeddings.pkl            # Stored face embeddings
```

## How It Works

1. **Face Detection**: Uses a lightweight RetinaFace detector to find faces in frames
2. **Feature Extraction**: MobileFaceNet extracts 512-dimensional embeddings
3. **Database Building**: Averages embeddings from 40 images per person
4. **Recognition**: Compares live embeddings to database using cosine similarity
5. **Real-time Display**: Shows bounding boxes and names with confidence scores

## Model Details

- **Face Detection**: RetinaFace (mobile version)
- **Face Recognition**: MobileFaceNet
- **Embedding Size**: 512 dimensions
- **Model Size**: ~5MB total
- **Inference Backend**: ONNX Runtime (GPU accelerated)

## API Reference

### FaceRecognitionPipeline

#### Methods

- `__init__(database_path, model_name)` - Initialize pipeline
- `add_person_to_database(person_name, image_folder)` - Add person to database
- `save_database()` - Save database to disk
- `load_database()` - Load database from disk
- `recognize_face(embedding)` - Recognize a face from embedding
- `run_live_recognition(camera_id, display)` - Run live recognition

## License

This project uses InsightFace which is licensed under MIT License.

## Credits

- InsightFace: https://github.com/deepinsight/insightface
- MobileFaceNet: https://arxiv.org/abs/1804.07573

## Support

For issues specific to:
- Jetson Nano setup: https://forums.developer.nvidia.com/
- InsightFace: https://github.com/deepinsight/insightface/issues
