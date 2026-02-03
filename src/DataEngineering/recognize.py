"""
Load k-NN model and recognize face in image: returns member_id or None.
Uses distance threshold and optional nearest-neighbor ratio to reject "unknown".
"""
import json
from collections import Counter
from pathlib import Path

import joblib
from feature_extraction import FaceFeatureExtractor


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_DISTANCE_THRESHOLD = 30.0  # Tune: higher = more permissive, lower = stricter
DEFAULT_RATIO_THRESHOLD = 1.0  # If dist_1st/dist_2nd > this, treat as ambiguous (reject). 1.0 = disabled.


class FaceRecognizer:
    def __init__(self, model_dir=None, model_path="face_landmarker.task", distance_threshold=None, ratio_threshold=None):
        model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.clf = joblib.load(model_dir / "knn_face_model.joblib")
        scaler_path = model_dir / "scaler.joblib"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        with open(model_dir / "label_to_member_id.json", "r", encoding="utf-8") as f:
            self.label_to_member_id = json.load(f)
        self.distance_threshold = distance_threshold if distance_threshold is not None else DEFAULT_DISTANCE_THRESHOLD
        self.ratio_threshold = ratio_threshold if ratio_threshold is not None else DEFAULT_RATIO_THRESHOLD
        self.extractor = FaceFeatureExtractor(model_path=model_path)
        self._k = min(self.clf.n_neighbors, len(self.clf._fit_X))

    def recognize(self, image):
        """
        Recognize face in BGR image (numpy array).
        Returns (member_id or None, distance or None).
        Uses k-neighbor majority vote, mean distance threshold, and nearest-neighbor ratio for ambiguous reject.
        """
        feat = self.extractor.get_landmark_features_from_image(image)
        if feat is None:
            return None, None
        feat = feat.reshape(1, -1)
        if self.scaler is not None:
            feat = self.scaler.transform(feat)
        dists, indices = self.clf.kneighbors(feat, n_neighbors=self._k)
        dists = dists[0]
        indices = indices[0]
        # Labels of k neighbors (indices into training set -> use clf._y)
        neighbor_labels = self.clf._y[indices]
        majority_label = Counter(neighbor_labels).most_common(1)[0][0]
        mean_distance = float(dists.mean())
        distance_first = float(dists[0])
        if mean_distance > self.distance_threshold:
            return None, mean_distance
        if self._k >= 2 and self.ratio_threshold < 1.0:
            ratio = distance_first / (float(dists[1]) + 1e-10)
            if ratio > self.ratio_threshold:
                return None, mean_distance
        member_id = self.label_to_member_id[int(majority_label)]
        return member_id, mean_distance

    def recognize_from_path(self, image_path):
        """Load image from path and return (member_id or None, distance or None)."""
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.recognize(image)


def get_recognizer(model_dir=None, model_path="face_landmarker.task", distance_threshold=None, ratio_threshold=None):
    """Convenience: return a FaceRecognizer instance."""
    return FaceRecognizer(
        model_dir=model_dir,
        model_path=model_path,
        distance_threshold=distance_threshold,
        ratio_threshold=ratio_threshold,
    )


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Recognize face in image")
    parser.add_argument("image", type=str, help="Path to image")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--face-model", type=str, default="face_landmarker.task")
    parser.add_argument("--distance-threshold", type=float, default=30.0)
    parser.add_argument("--ratio-threshold", type=float, default=1.0, help="Reject if nearest/2nd-nearest distance ratio > this (1.0 = disabled)")
    args = parser.parse_args()
    r = get_recognizer(model_dir=args.model_dir, model_path=args.face_model, distance_threshold=args.distance_threshold, ratio_threshold=args.ratio_threshold)
    member_id, distance = r.recognize_from_path(args.image)
    if member_id:
        print(f"Member: {member_id} (distance={distance:.2f})")
    else:
        print(f"Unknown (distance={distance})" if distance is not None else "No face detected")
    sys.exit(0 if member_id else 1)
