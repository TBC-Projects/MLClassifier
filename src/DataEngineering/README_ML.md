# Face recognition and attendance (custom ML)

Custom k-NN face recognizer using MediaPipe landmarks (no pretrained face recognition models). Recognized members can be logged as present.

## Setup

```bash
pip3 install -r requirements.txt
```

## How to start running

1. Install dependencies (see Setup above).
2. Add or configure members in `members.json` (see **Managing members** below).
3. Train the model once (see **How to train**).
4. Run webcam attendance or single-image recognition (see **Recognition and attendance**).

**Commands in order:**

```bash
python3 train.py --out-dir models
python3 webcam_attendance.py --model-dir models --attendance attendance.json
```

After you add or change member folders, re-run training before running attendance again.

## How to train

From the project root (FaceML):

```bash
python train.py --out-dir models
```

Optional: `--members path/to/members.json`, `--model face_landmarker.task`, `--k 5`, `--tune-threshold` (suggest a distance threshold from validation accuracy).

Training builds normalized landmark features from each member's folder, standardizes them, fits a k-NN classifier, and writes:

- `models/knn_face_model.joblib`
- `models/scaler.joblib`
- `models/label_to_member_id.json`

Use `--tune-threshold` after training to print a suggested `--distance-threshold` for recognition.

## Managing members

Each member in `members.json` has:

- **member_id** – internal id (e.g. `person2`)
- **name** – name shown when the person is detected (e.g. "Alice Smith")
- **folder** – folder name under the project containing that member's face images (e.g. `person2`)

Example entry:

```json
{"member_id": "person2", "name": "Alice Smith", "folder": "person2"}
```

### Add a new member's folder

1. **Capture face images** for the new person, or create the folder and add images manually (multiple photos per person, same style as `person2`–`person5`).
   - To capture aligned faces from the webcam into a person folder (e.g. `person7`):
     ```bash
     python3 face_extraction_mine_trial.py --capture --person person7 --num-captures 20
     ```
     This creates `person7/` and saves `picture1.jpg`, `picture2.jpg`, … (aligned, 160×160). Without `--person`, images go to `--output-folder` (default `photosML`) as `face_001.jpg`, `face_002.jpg`, …
   - Optional: `--num-captures N`, `--output-folder FOLDER` (used only when `--person` is not set).
2. Add one entry to `members.json` with `member_id`, `name`, and `folder` (the new folder name):

   ```json
   {"member_id": "person6", "name": "New Member", "folder": "person6"}
   ```

3. Re-run training so the new member is included:

   ```bash
   python train.py --out-dir models
   ```

**Capture script reference** (`face_extraction_mine_trial.py`):

| Option | Description |
|--------|-------------|
| `--capture` | Run aligned face capture from webcam |
| `--person FOLDER` | Save to this folder as `picture1.jpg`, `picture2.jpg`, … (e.g. `person7`) |
| `--num-captures N` | Number of images to capture (default: 20) |
| `--output-folder FOLDER` | Used when `--person` is not set (default: `photosML`); files named `face_001.jpg`, … |

### Rename a folder to a member's name

1. Rename the folder on disk (e.g. `person2` → `alice_smith`).
2. In `members.json`, set that member's `folder` to the new folder name (e.g. `"folder": "alice_smith"`).
3. Re-run training:

   ```bash
   python train.py --out-dir models
   ```

### Change the name displayed when a person is detected

Edit only the `name` field for that member in `members.json`. You do not need to re-train or rename folders; the display updates the next time you run webcam attendance or recognition.

## Recognition and attendance

**Single image:**

```bash
python recognize.py path/to/image.jpg --model-dir models
```

**Webcam attendance:**

```bash
python webcam_attendance.py --model-dir models --attendance attendance.json
```

The window shows "Member: &lt;name&gt;" (using the `name` from `members.json`), "Not in club" (face detected but not a member), or "No face" (no face detected). Recognized members are logged to the attendance file (debounced so the same person is not logged every frame). Optional flags: `--debounce 30`, `--distance-threshold 30` (higher = more permissive), `--ratio-threshold 1.0` (1.0 = disabled; lower values reject ambiguous matches).

## Files added (face_extraction.py unchanged)

| File | Purpose |
|------|--------|
| `face_extraction_mine_trial.py` | Capture aligned faces from webcam; use `--person FOLDER` to save new member data as `picture1.jpg`, … |
| `requirements.txt` | Dependencies |
| `members.json` | Member list (id, name, folder) |
| `feature_extraction.py` | MediaPipe → normalized landmark vector |
| `member_db.py` | Load/save members.json |
| `train.py` | Build dataset and train k-NN |
| `recognize.py` | Load model and recognize(image) |
| `attendance.py` | Log attendance with debouncing |
| `webcam_attendance.py` | Webcam loop: recognize + log attendance |
