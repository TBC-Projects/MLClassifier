import cv2
import torch
import numpy as np
from scipy import ndimage

# Configuration
DEFAULT_IMAGE_SIZE = (160, 160)

def augment_face(img):
    """Apply random augmentations for training"""
    # Random rotation (-10 to 10 degrees)
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10, 10)
        img = ndimage.rotate(img, angle, reshape=False, mode='nearest')
    
    # Random brightness
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.7, 1.3)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)
    
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    
    # Random noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def preprocess_face(img, augment=False, target_size=DEFAULT_IMAGE_SIZE):
    """Preprocess face image with optional augmentation"""
    if img is None or img.size == 0:
        raise ValueError("Invalid image")
    
    # Apply augmentation if training
    if augment:
        img = augment_face(img)
    
    # Resize to 160×160
    img = cv2.resize(img, target_size)
    
    # Histogram equalization for better contrast
    img = cv2.equalizeHist(img)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor [1, 160, 160] - only ONE unsqueeze!
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    
    return img

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)