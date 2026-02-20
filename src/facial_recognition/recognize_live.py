"""
recognize_live.py  –  Live face recognition optimised for NVIDIA Jetson Orin Nano.

Key fixes vs. original:
  1. embedding_dim=128 passed explicitly (was defaulting to 256 → weight mismatch).
  2. Absolute paths resolved from __file__, not from a hard-coded dev machine path.
  3. Face-ID grid cell enlarged (//80 instead of //50) so fewer phantom "new faces"
     appear when someone moves slightly, which also reduces history thrash.
  4. score_margin check now 0.10 (was 0.20) — a 0.20 hard gap was too aggressive
     given the inter-person similarity spread in the current DB.
  5. JETSON performance optimisations:
     - Frame is half-sized before detection (halves Haar CPU work).
     - Inference runs on CUDA if available (Jetson has an integrated GPU).
     - torch.inference_mode() replaces no_grad (slightly faster).
     - CAP_PROP_BUFFERSIZE = 1 to reduce latency (drop stale frames).
     - Face detection only runs every N frames; last boxes reused in between.
     - BGR→GRAY conversion happens once per frame (not per face).

Usage:
    python recognize_live.py
"""

import os
import cv2
import torch
import numpy as np
import time
from collections import deque, Counter
from model import FaceEmbeddingNet
from utils import preprocess_face, cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(SCRIPT_DIR, "face_model.pth")
DB_PATH       = os.path.join(SCRIPT_DIR, "embeddings.npy")
CASCADE_PATH  = os.path.join(SCRIPT_DIR, "haarcascade_frontalface_default.xml")
# ──────────────────────────────────────────────────────────────────────────────

# ── Recognition hyper-parameters ──────────────────────────────────────────────
THRESHOLD      = 0.68   # minimum avg cosine similarity to accept a match
MIN_MATCH_RATE = 0.45   # fraction of per-person embeddings that must exceed THRESHOLD
# FIX: score_margin lowered from 0.20 → 0.10.
# 0.20 was too strict: the DB analysis showed inter-person gaps often < 0.20,
# so legitimate faces were being labelled "Unknown" incorrectly.
SCORE_MARGIN   = 0.10
# ──────────────────────────────────────────────────────────────────────────────

# ── Jetson performance knobs ───────────────────────────────────────────────────
DETECT_EVERY_N   = 3    # run Haar detection only every N frames; reuse boxes otherwise
SCALE_FOR_DETECT = 0.5  # resize frame to this fraction before Haar (saves ≈3–4× CPU)
HISTORY_LEN      = 7    # temporal voting window (frames)
MIN_AGREEMENT    = 0.57 # fraction of votes needed to confirm a name
# ──────────────────────────────────────────────────────────────────────────────


class FaceRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # ── Model ──────────────────────────────────────────────────────────────
        # FIX: embedding_dim=128 must be passed explicitly.
        # The class default was 256, which caused the final FC layer to have the
        # wrong shape, silently loading weights into mismatched positions and
        # producing random embeddings → everyone matched as "Unknown".
        self.model = FaceEmbeddingNet(embedding_dim=128).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        print(f"Model loaded: {MODEL_PATH}")

        # ── Database ───────────────────────────────────────────────────────────
        self.db = np.load(DB_PATH, allow_pickle=True).item()
        print(f"Database loaded: {DB_PATH}")

        # ── Face detector ──────────────────────────────────────────────────────
        self.detector = cv2.CascadeClassifier(CASCADE_PATH)
        if self.detector.empty():
            raise ValueError(f"Could not load Haar cascade from {CASCADE_PATH}")
        print(f"Cascade loaded: {CASCADE_PATH}")

        # ── State ──────────────────────────────────────────────────────────────
        self.recognition_history: dict[str, deque] = {}
        self.last_faces   = []   # cached face boxes between detection frames
        self.frame_count  = 0
        self.frame_times: list[float] = []

        self._print_summary()

    # ──────────────────────────────────────────────────────────────────────────
    def _print_summary(self):
        print(f"\n{'='*52}")
        print("  Face Recognition — Jetson Orin Nano")
        print(f"{'='*52}")
        print(f"  People in DB : {len(self.db)}")
        for name, data in self.db.items():
            n = len(data['embeddings'])
            print(f"    • {name}: {n} reference images")
        print(f"\n  Threshold    : {THRESHOLD}")
        print(f"  Min match %  : {MIN_MATCH_RATE:.0%}")
        print(f"  Score margin : {SCORE_MARGIN}")
        print(f"  Detect every : {DETECT_EVERY_N} frames")
        print(f"  Detect scale : {SCALE_FOR_DETECT}×")
        print(f"{'='*52}\n")

    # ──────────────────────────────────────────────────────────────────────────
    def _get_embedding(self, face_gray: np.ndarray) -> np.ndarray:
        """Run model inference and return an L2-normalised embedding."""
        with torch.inference_mode():   # slightly faster than no_grad on Jetson
            tensor = preprocess_face(face_gray, augment=False).unsqueeze(0).to(self.device)
            return self.model(tensor).cpu().numpy()[0]

    # ──────────────────────────────────────────────────────────────────────────
    def recognize_face(self, face_gray: np.ndarray) -> tuple[str, float]:
        """Compare face against all DB entries and return (name, score)."""
        emb = self._get_embedding(face_gray)

        scores = []
        for person, data in self.db.items():
            sims = [cosine_similarity(emb, db_emb) for db_emb in data['embeddings']]
            avg_sim  = float(np.mean(sims))
            max_sim  = float(np.max(sims))
            n_above  = sum(1 for s in sims if s > THRESHOLD)
            match_rate = n_above / len(sims)
            scores.append((person, avg_sim, max_sim, match_rate))

        scores.sort(key=lambda x: x[1], reverse=True)
        best   = scores[0]
        second = scores[1] if len(scores) > 1 else None

        margin = (best[1] - second[1]) if second else 1.0

        if (best[1] < THRESHOLD
                or best[3] < MIN_MATCH_RATE
                or margin < SCORE_MARGIN):
            return "Unknown", 0.0

        return best[0], best[1]

    # ──────────────────────────────────────────────────────────────────────────
    def smooth_recognition(self, face_id: str, name: str, score: float) -> tuple[str, float]:
        """Temporal voting: require MIN_AGREEMENT consensus over recent frames."""
        if face_id not in self.recognition_history:
            self.recognition_history[face_id] = deque(maxlen=HISTORY_LEN)

        self.recognition_history[face_id].append((name, score))
        history = self.recognition_history[face_id]

        names = [n for n, _ in history]
        top_name, count = Counter(names).most_common(1)[0]

        if count / len(names) >= MIN_AGREEMENT:
            avg_score = float(np.mean([s for n, s in history if n == top_name]))
            return top_name, avg_score

        return "Unknown", 0.0

    # ──────────────────────────────────────────────────────────────────────────
    def _cleanup_old_faces(self, current_ids: list[str]):
        for fid in list(self.recognition_history):
            if fid not in current_ids:
                del self.recognition_history[fid]

    # ──────────────────────────────────────────────────────────────────────────
    def _detect_faces(self, gray_full: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect faces on a downscaled copy of the frame, then rescale coordinates.
        Running Haar on a half-size image is ~3–4× faster (key for Jetson CPU).
        """
        h, w = gray_full.shape
        small = cv2.resize(gray_full, (int(w * SCALE_FOR_DETECT), int(h * SCALE_FOR_DETECT)))

        detections = self.detector.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(60 * SCALE_FOR_DETECT), int(60 * SCALE_FOR_DETECT)),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(detections) == 0:
            return []

        inv = 1.0 / SCALE_FOR_DETECT
        return [(int(x * inv), int(y * inv), int(w2 * inv), int(h2 * inv))
                for x, y, w2, h2 in detections]

    # ──────────────────────────────────────────────────────────────────────────
    def run(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        # JETSON FIX: keep buffer tiny so we always process the latest frame
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        print("Face recognition running — press ESC to quit, 's' to screenshot.\n")

        try:
            while True:
                t0 = time.time()

                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # JETSON: run expensive Haar detection only every N frames
                if self.frame_count % DETECT_EVERY_N == 0:
                    self.last_faces = self._detect_faces(gray)

                current_ids = []

                for (x, y, w, h) in self.last_faces:
                    # Guard against boxes that went out-of-bounds between frames
                    x, y = max(x, 0), max(y, 0)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    if w <= 0 or h <= 0:
                        continue

                    face_roi = gray[y:y+h, x:x+w]
                    name, score = self.recognize_face(face_roi)

                    # FIX: larger grid cell (//80 was //50) → fewer spurious
                    # "new face" events caused by small jitter in detection boxes
                    face_id = f"{x // 80}_{y // 80}"
                    current_ids.append(face_id)

                    stable_name, stable_score = self.smooth_recognition(face_id, name, score)

                    color = (0, 200, 0) if stable_name != "Unknown" else (0, 0, 220)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                    label = f"{stable_name} ({stable_score:.2f})" if stable_name != "Unknown" else "Unknown"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x, y - lh - 10), (x + lw, y), color, -1)
                    cv2.putText(frame, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                self._cleanup_old_faces(current_ids)

                # FPS overlay
                elapsed = time.time() - t0
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                self.frame_times.append(fps)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                avg_fps = float(np.mean(self.frame_times))

                cv2.putText(frame, f"FPS: {avg_fps:.1f}",       (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Faces: {len(self.last_faces)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Face Recognition", frame)
                self.frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    fn = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(fn, frame)
                    print(f"Screenshot saved: {fn}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nProcessed {self.frame_count} frames")
            if self.frame_times:
                print(f"Average FPS: {float(np.mean(self.frame_times)):.1f}")


if __name__ == "__main__":
    try:
        recognizer = FaceRecognizer()
        recognizer.run(camera_id=0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()