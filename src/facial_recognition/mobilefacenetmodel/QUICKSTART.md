# Quick Start Guide - Face Recognition on Jetson Nano

## 🚀 5-Minute Setup

### Step 1: Installation (5 minutes)

```bash
# Make installation script executable
chmod +x install_jetson.sh

# Run installation
./install_jetson.sh

# Wait for installation to complete
```

### Step 2: Prepare Training Data (5 minutes)

Create this folder structure:

```
training_data/
├── alice/          # 40 images of Alice
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── bob/            # 40 images of Bob
│   ├── img001.jpg
│   └── ...
├── charlie/        # 40 images of Charlie
│   └── ...
└── diana/          # 40 images of Diana
    └── ...
```

### Step 3: Build Database (2 minutes)

```bash
python3 face_recognition_pipeline.py
```

When prompted:
- Type `y` and press Enter
- Type `training_data` (or your folder path)
- Wait for processing (~2 minutes for 160 images)

### Step 4: Run Live Recognition

The system starts automatically after building the database!

Press `q` to quit.

---

## 📊 What to Expect

### Performance Metrics
- **FPS**: 15-25 frames per second
- **Accuracy**: >95% for well-trained faces
- **Latency**: <100ms per frame

### First Run
- Model download: ~30 seconds (one-time)
- Database building: ~1-2 minutes
- Recognition starts immediately after

---

## 🎯 Quick Commands

### Build Database Only
```bash
python3 database_manager.py
# Choose option 3: Batch add from training data folder
```

### Test with Single Image
```bash
python3 test_pipeline.py
# Choose option 1: Test with a single image
```

### View Database
```bash
python3 database_manager.py
# Choose option 1: List all people
```

### Maximum Performance Mode
```bash
# Enable max performance (run before starting recognition)
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## 🔧 Common Issues & Quick Fixes

### Issue: Low FPS (< 10)

**Solution:**
```bash
# Enable max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Then edit face_recognition_pipeline.py:
# Change line ~198: frame_skip = 3  # was 2
```

### Issue: Camera not found

**Solution:**
```bash
# List available cameras
ls /dev/video*

# Try different camera IDs in the script
# Edit main() function, change: camera_id=0 to camera_id=1
```

### Issue: Poor recognition accuracy

**Solutions:**
1. Add more varied training images (different angles, lighting)
2. Lower threshold in `face_recognition_pipeline.py`:
   ```python
   self.threshold = 0.5  # was 0.6
   ```
3. Increase detection size:
   ```python
   det_size=(640, 640)  # was (320, 320)
   ```

### Issue: Out of memory

**Solution:**
```bash
# Close other applications
# Reduce detection size in face_recognition_pipeline.py:
det_size=(240, 240)  # was (320, 320)
```

---

## 📁 File Guide

| File | Purpose | When to Use |
|------|---------|-------------|
| `face_recognition_pipeline.py` | Main system | Run this to start recognition |
| `database_manager.py` | Manage database | Add/remove people |
| `test_pipeline.py` | Test system | Verify before live run |
| `install_jetson.sh` | Installation | Run once on setup |

---

## 🎓 Tips for Best Results

### Training Images
✅ **DO:**
- Use clear, well-lit photos
- Include variety (happy, neutral, serious)
- Capture different angles (±15 degrees)
- Use consistent quality images

❌ **DON'T:**
- Use blurry or dark images
- Include multiple people in frame
- Use extreme angles or occlusions
- Mix high and low quality images

### Live Recognition
✅ **DO:**
- Ensure good lighting
- Face camera directly
- Stay within 1-3 meters
- Keep face clearly visible

❌ **DON'T:**
- Use in very dark environments
- Wear masks or heavy glasses
- Stand too far or too close
- Move too quickly

---

## 🔄 Updating the Database

### Add New Person
```bash
python3 database_manager.py
# Choose: 2. Add person from folder
# Enter name and folder path
```

### Remove Person
```bash
python3 database_manager.py
# Choose: 4. Remove person
# Enter name to remove
```

### Rebuild Everything
```bash
python3 database_manager.py
# Choose: 6. Clear database
# Then run main script again to rebuild
```

---

## 📈 Performance Tuning

### Preset Configurations

**Balanced (Default)**
```python
det_size=(320, 320)
frame_skip = 2
# Expected: 15-20 FPS, High accuracy
```

**Speed Priority**
```python
det_size=(240, 240)
frame_skip = 3
# Expected: 25-30 FPS, Good accuracy
```

**Quality Priority**
```python
det_size=(640, 640)
frame_skip = 1
# Expected: 8-12 FPS, Very high accuracy
```

Edit these values in `face_recognition_pipeline.py` around lines 25-27 and 198.

---

## 🆘 Getting Help

### Check System Status
```bash
# Check GPU usage
sudo tegrastats

# Check CUDA
nvcc --version

# Check Python packages
pip3 list | grep -E "insightface|opencv|onnx"
```

### Useful Debug Commands
```bash
# Test camera
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink

# Check model files
ls -lh ~/.insightface/models/

# View database
python3 -c "from face_recognition_pipeline import *; p = FaceRecognitionPipeline(); print(f'People: {list(p.face_database.keys())}')"
```

---

## ✅ Verification Checklist

Before running live recognition:

- [ ] Installation completed successfully
- [ ] Training images organized in folders
- [ ] Database built (check for `face_database/embeddings.pkl`)
- [ ] Camera connected and detected
- [ ] Max performance mode enabled (optional but recommended)
- [ ] Tested with single image (optional)

---

## 🎉 You're Ready!

Run:
```bash
python3 face_recognition_pipeline.py
```

And start recognizing faces in real-time!

---

## 📞 Support Resources

- **Jetson Nano Forums**: https://forums.developer.nvidia.com/
- **InsightFace Docs**: https://github.com/deepinsight/insightface
- **OpenCV Tutorials**: https://docs.opencv.org/

---

**Happy Face Recognizing! 🎭**
