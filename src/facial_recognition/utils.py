import cv2
import torch
import numpy as np
from scipy import ndimage

# Configuration
DEFAULT_IMAGE_SIZE = (160, 160)


def augment_face(img):
    """Apply random augmentations for training."""
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10, 10)
        img = ndimage.rotate(img, angle, reshape=False, mode='nearest')

    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.7, 1.3)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)

    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img


def preprocess_face(img, augment=False, target_size=DEFAULT_IMAGE_SIZE):
    """
    Preprocess a grayscale face crop for the model.

    Returns a tensor of shape [1, H, W].
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid image: None or empty")

    if augment:
        img = augment_face(img)

    img = cv2.resize(img, target_size)
    img = cv2.equalizeHist(img)
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return img


def cosine_similarity(a, b):
    """Cosine similarity between two numpy vectors. Returns value in [-1, 1]."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))