"""
build_database.py  –  Build the face-embedding database from a dataset folder.

Usage (run from the project root):
    python build_database.py

Dataset layout expected:
    dataset/
        Alice/
            img1.jpg
            img2.jpg
            ...
        Bob/
            img1.jpg
            ...

Outputs:
    embeddings.npy   (dict: name → {'embeddings': ndarray, 'mean': ndarray})
"""

import os
import cv2
import torch
import numpy as np
from model import FaceEmbeddingNet
from utils import preprocess_face

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "face_model.pth")
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, "embeddings.npy")
# ──────────────────────────────────────────────────────────────────────────────


def build_database():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # FIX: embedding_dim=128 must match trained weights.
    # The default in model.py was incorrectly 256, causing garbage embeddings.
    model = FaceEmbeddingNet(embedding_dim=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")

    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

    db = {}

    with torch.no_grad():
        for person in sorted(os.listdir(DATASET_PATH)):
            person_path = os.path.join(DATASET_PATH, person)
            if not os.path.isdir(person_path):
                continue

            embs = []
            for img_name in sorted(os.listdir(person_path)):
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, 0)   # grayscale

                if img is None:
                    print(f"  Warning: Could not load {img_path}")
                    continue

                try:
                    face_tensor = preprocess_face(img, augment=False).unsqueeze(0).to(device)
                    emb = model(face_tensor).cpu().numpy()[0]
                    embs.append(emb)
                except Exception as e:
                    print(f"  Warning: Failed to process {img_path}: {e}")

            if embs:
                db[person] = {
                    'embeddings': np.array(embs),          # (N, 128)
                    'mean':       np.mean(embs, axis=0),   # (128,)
                }
                print(f"  Added '{person}': {len(embs)} embeddings")
            else:
                print(f"  Warning: No valid images for '{person}' — skipped")

    np.save(OUTPUT_PATH, db, allow_pickle=True)
    print(f"\nDatabase saved to {OUTPUT_PATH} with {len(db)} people.")


if __name__ == "__main__":
    build_database()