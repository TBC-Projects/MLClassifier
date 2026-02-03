"""
Extract normalized face landmark features for ML (training and inference).
Uses MediaPipe Face Landmarker; no pretrained face recognition models.
"""
import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Landmark indices (MediaPipe Face Landmarker has 478 landmarks)
NOSE_TIP_IDX = 1
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263


class FaceFeatureExtractor:
    """Extract normalized landmark feature vectors from images using MediaPipe."""

    def __init__(self, model_path="face_landmarker.task", num_faces=1):
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def _landmarks_to_vector(self, face_landmarks):
        """Convert MediaPipe face_landmarks to numpy array (478, 3)."""
        n = len(face_landmarks)
        arr = np.array([[p.x, p.y, p.z] for p in face_landmarks], dtype=np.float32)
        return arr

    def _normalize_landmarks(self, landmarks):
        """
        Center by nose tip, rotate so eye line is horizontal (rotation-invariant),
        then scale by inter-eye distance so the same person under different
        pose/scale/head-tilt maps to similar vectors.
        """
        nose = landmarks[NOSE_TIP_IDX]
        left_eye = landmarks[LEFT_EYE_IDX]
        right_eye = landmarks[RIGHT_EYE_IDX]
        # Inter-eye vector (2D) for scale and rotation
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        scale = np.sqrt(dx * dx + dy * dy)
        if scale < 1e-6:
            scale = 1.0
        # Center
        centered = landmarks - nose
        # Rotate (x, y) so eye line is horizontal; leave z unchanged
        theta = np.arctan2(dy, dx)
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        xy = centered[:, :2].copy()
        centered[:, 0] = xy[:, 0] * cos_t - xy[:, 1] * sin_t
        centered[:, 1] = xy[:, 0] * sin_t + xy[:, 1] * cos_t
        scaled = centered / scale
        return scaled.flatten()

    def get_landmark_features_from_image(self, image):
        """
        Extract normalized feature vector from a BGR image (numpy array).
        Returns shape (478*3,) or None if no face detected.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = self.face_landmarker.detect(mp_image)
        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return None
        landmarks = self._landmarks_to_vector(detection_result.face_landmarks[0])
        return self._normalize_landmarks(landmarks)

    def get_landmark_features_from_path(self, image_path):
        """
        Load image from path and return normalized feature vector or None.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.get_landmark_features_from_image(image)


def get_feature_extractor(model_path="face_landmarker.task"):
    """Convenience: return a FaceFeatureExtractor instance."""
    return FaceFeatureExtractor(model_path=model_path)
