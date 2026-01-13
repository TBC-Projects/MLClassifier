import os
import time
import cv2
import numpy as np
import joblib
from insightface.app import FaceAnalysis
from sklearn.svm import SVC
from collections import defaultdict

class FaceRecognizer:
    """
    Face recognition engine for pre-aligned grayscale images
    - Uses InsightFace embeddings
    - Trains SVM classifier
    - Supports unknown detection
    Optimized for Jetson Nano
    """

    def __init__(self, model_name="buffalo_s", ctx_id=0, threshold=0.6):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=(160, 160))
        self.svm = None
        self.threshold = threshold

    # ----------------------------
    # EMBEDDING
    # ----------------------------
    def get_embedding(self, image: np.ndarray):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        faces = self.app.get(image, max_num=1)
        if len(faces) == 0:
            return None
        return faces[0].embedding

    # ----------------------------
    # TRAINING
    # ----------------------------
    def train(self, dataset_path: str):
        embeddings = defaultdict(list)
        for person in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person)
            if not os.path.isdir(person_dir):
                continue

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                emb = self.get_embedding(img)
                if emb is not None:
                    embeddings[person].append(emb)

        X, y = [], []
        for label, embs in embeddings.items():
            X.append(np.mean(embs, axis=0))
            y.append(label)

        self.svm = SVC(kernel="linear", probability=True)
        self.svm.fit(np.array(X), np.array(y))

    # ----------------------------
    # PREDICTION
    # ----------------------------
    def predict(self, image: np.ndarray):
        emb = self.get_embedding(image)
        if emb is None:
            return "No face", None

        probs = self.svm.predict_proba(emb.reshape(1, -1))[0]
        idx = np.argmax(probs)
        label = self.svm.classes_[idx]
        confidence = probs[idx]

        if confidence >= self.threshold:
            return label, confidence
        else:
            return "Unknown", confidence

    # ----------------------------
    # SAVE / LOAD
    # ----------------------------
    def save(self, path="svm_model.pkl"):
        if self.svm is not None:
            joblib.dump(self.svm, path)

    def load(self, path="svm_model.pkl"):
        self.svm = joblib.load(path)


# ----------------------------
# BACKGROUND PROCESSING LOOP
# ----------------------------
def monitor_folder(folder_path, recognizer: FaceRecognizer, processed_folder="processed", poll_interval=2):
    """
    Continuously monitor a folder for new grayscale images and classify them.
    :param folder_path: folder to monitor
    :param recognizer: FaceRecognizer instance (SVM already trained or loaded)
    :param processed_folder: folder to move processed images
    :param poll_interval: seconds between folder scans
    """
    os.makedirs(processed_folder, exist_ok=True)
    seen_files = set()

    while True:
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and fname not in seen_files:
                file_path = os.path.join(folder_path, fname)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                label, confidence = recognizer.predict(img)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {fname} -> {label} ({confidence:.2f})")

                # Move processed image
                dest_path = os.path.join(processed_folder, fname)
                os.rename(file_path, dest_path)
                seen_files.add(fname)

        time.sleep(poll_interval)
