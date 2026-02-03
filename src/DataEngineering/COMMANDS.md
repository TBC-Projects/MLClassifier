# Face Extraction — Command Reference

This document lists all available commands and options for `face_extraction.py`.

---

## Basic usage

### 1. Image-based face extraction

Extract face regions and landmarks from a static image.

**Command:**

```bash
python face_extraction.py --image <image_path> [--output <file_path>] [--output-folder <folder>]
```

| Option | Description |
|--------|-------------|
| `--image` | Path to input image (required) |
| `--output` | Optional path to save annotated output image file (if not specified, image is displayed in a window) |
| `--output-folder` | Optional folder to save output image (saves with original filename + `_annotated` suffix). Creates folder if it doesn't exist. |

**Examples:**

```bash
python face_extraction.py --image photo1.jpg
python face_extraction.py --image photo1.jpg --output result.jpg
python face_extraction.py --image photo1.jpg --output-folder results/
```

---

### 2. Webcam face detection

Real-time face detection and tracking from webcam feed.

- Tracks up to 3 faces simultaneously
- Color-coded: Face 1 (Green), Face 2 (Blue), Face 3 (Red)
- Shows bounding boxes around detected faces

**Command:**

```bash
python face_extraction.py --webcam [--camera <index>]
```

| Option | Description |
|--------|-------------|
| `--webcam` | Enable webcam mode |
| `--camera` | Camera index (default: 0). Use 1 for second webcam, 2 for third, etc. |

**Controls:**

- Press **q** to quit  
- Press **ESC** to quit  
- Close the window to quit  

**Examples:**

```bash
python face_extraction.py --webcam
python face_extraction.py --webcam --camera 1
```

---

### 3. Face dataset capture

Capture face images from webcam for dataset creation.

- Saves images as 160×160 JPG (square crop, no distortion)
- Only captures the first detected face
- Images saved as `picture1.jpg`, `picture2.jpg`, etc.
- Optional manual mode: press **SPACE** to take each picture
- On-screen pose instructions (phases: close-up, angles, 3/4 view, resting face)

**Command:**

```bash
python face_extraction.py --capture [--person <folder_name>] [--num-images <number>] [--camera <index>] [--manual]
```

| Option | Description |
|--------|-------------|
| `--capture` | Enable face dataset capture mode |
| `--person` | Output folder name to save images (default: `person1`). Creates folder if it doesn't exist. |
| `--num-images` | Number of images to capture (default: 20) |
| `--camera` | Camera index (default: 0). Use 1 for second webcam. |
| `--manual` | Manual trigger mode. Capture only when SPACE is pressed. (Default: automatic, one image per second when face detected.) |

**Behavior (automatic mode):**

- 5-second countdown, then captures one image every 1 second when face is detected
- Press **q** or **ESC** to stop early

**Behavior (manual mode, `--manual`):**

- No auto-capture. Press **SPACE** to take each picture when a face is detected.
- On-screen instructions guide poses by phase (close-up → slight tilts → angles → resting face).
- Press **q** or **ESC** to stop early

**Examples:**

```bash
python face_extraction.py --capture
python face_extraction.py --capture --person person2
python face_extraction.py --capture --person person1 --num-images 30
python face_extraction.py --capture --person JohnDoe --num-images 40 --camera 1 --manual
```

---

### 4. Process existing images

Process existing images and extract face regions with bounding boxes.

- Can process a single image or all images in a folder
- Extracts face regions (bounding box area) from each image
- Supports multiple faces per image
- Automatically resizes to 160×160 (can be disabled)

**Command:**

```bash
python face_extraction.py --process <image_or_folder> [--process-output <folder>] [--no-resize]
```

| Option | Description |
|--------|-------------|
| `--process` | Path to image file or folder containing images (required) |
| `--process-output` | Output folder name for processed faces (default: `processed_faces`) |
| `--no-resize` | Keep original face size instead of resizing to 160×160 |

**Behavior:**

- Processes all images in folder (jpg, jpeg, png, bmp, tiff, webp)
- Extracts all detected faces from each image
- Multiple faces in one image: saved as `filename_face1.jpg`, `filename_face2.jpg`, etc.
- Single face in image: saved with original filename
- Adds padding around face for better cropping
- Resizes to 160×160 by default (use `--no-resize` to keep original size)

**Examples:**

```bash
python face_extraction.py --process person1/
python face_extraction.py --process photo1.jpg
python face_extraction.py --process person1/ --process-output extracted_faces
python face_extraction.py --process photo1.jpg --no-resize
```

---

## Condensed command list

```bash
# Image extraction (display in window)
python face_extraction.py --image photo.jpg

# Image extraction (save to specific file)
python face_extraction.py --image photo.jpg --output result.jpg

# Image extraction (save to folder)
python face_extraction.py --image photo.jpg --output-folder results/

# Webcam detection (real-time, up to 3 faces)
python face_extraction.py --webcam

# Webcam with second camera
python face_extraction.py --webcam --camera 1

# Face dataset capture (default: 20 images, person1 folder)
python face_extraction.py --capture

# Face dataset capture (custom folder)
python face_extraction.py --capture --person person2

# Face dataset capture (custom folder and image count)
python face_extraction.py --capture --person person3 --num-images 30

# Face dataset capture with second camera
python face_extraction.py --capture --person person1 --num-images 20 --camera 1

# Manual capture (press SPACE to take each picture, 40 images to folder JohnDoe)
python face_extraction.py --capture --person JohnDoe --num-images 40 --camera 1 --manual

# Process existing images from folder (resize to 160x160)
python face_extraction.py --process person1/

# Process single image (keep original size)
python face_extraction.py --process photo1.jpg --no-resize

# Process folder with custom output directory
python face_extraction.py --process person1/ --process-output extracted_faces
```

---

## Requirements

Before running any commands, activate the conda environment:

```bash
conda activate mediapipe_env
```

Or use the helper scripts:

- `run_webcam.bat` (for webcam mode)
- `run_webcam.ps1` (PowerShell script for webcam mode)

---

## Output locations

| Mode | Output |
|------|--------|
| Image extraction | Specified by `--output` (file path) or `--output-folder` (folder path). If neither specified, image is displayed in window. |
| Webcam detection | Real-time display only (no saving) |
| Dataset capture | Saves to `<person_folder>/picture1.jpg`, `picture2.jpg`, etc. Default: `person1/` in project directory. Customize with `--person`. |
| Process existing images | Saves to `<process-output>/` folder. Default: `processed_faces/`. Customize with `--process-output`. |

---

## Notes

1. The `face_landmarker.task` model file must be in the project directory.
2. **Camera index:** 0 = first (default), 1 = second webcam. Use `--camera` to switch.
3. Dataset capture only saves the first detected face if multiple people are present.
4. All images are saved in JPG format.
5. Dataset images are square-cropped and resized to 160×160 (no face distortion).
6. Face detection supports up to 3 faces simultaneously in webcam mode.
7. Process existing images can extract multiple faces from a single image.
8. Supported image formats: JPG, JPEG, PNG, BMP, TIFF, WEBP.
9. With `--manual`, capture runs until you press SPACE for each image (no 1/sec auto-capture).
10. Capture mode shows on-screen pose instructions (with background) when using manual or auto.

---

*Last updated: Based on face_extraction.py (camera, manual, instructions, square crop).*
