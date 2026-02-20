"""
train.py  –  Train FaceEmbeddingNet with triplet loss.

Usage (run from the project root):
    python train.py

Dataset layout:
    dataset/
        Alice/   (≥ 2 images)
        Bob/     (≥ 2 images)
        ...
"""

import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import FaceEmbeddingNet
from utils import preprocess_face

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")
MODEL_SAVE   = os.path.join(SCRIPT_DIR, "face_model.pth")
# ──────────────────────────────────────────────────────────────────────────────

# ── Hyper-parameters ───────────────────────────────────────────────────────────
EMBEDDING_DIM = 128     # FIX: was 256 in old model.py; must match here and in build_database
BATCH_SIZE    = 24
EPOCHS        = 100
LR            = 1e-3
MARGIN        = 0.3     # FIX: raised from 0.2 → 0.3 for more separation between people
PATIENCE      = 15      # early-stopping patience (epochs)
# ──────────────────────────────────────────────────────────────────────────────


def triplet_loss(anchor, positive, negative, margin=MARGIN):
    """
    Online triplet loss (all triplets are semi-hard by construction).
    A larger margin forces the model to push negatives further away,
    which helps distinguish similar-looking people.
    """
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()


class TripletDataset(Dataset):
    def __init__(self, dataset_path, augment=False):
        self.augment = augment

        self.person_images: dict[str, list[str]] = {}

        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path):
                continue
            images = [
                os.path.join(person_path, f)
                for f in os.listdir(person_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ]
            if len(images) >= 2:
                self.person_images[person] = images

        self.people = list(self.person_images.keys())

        if len(self.people) < 2:
            raise ValueError(
                f"Need ≥ 2 people in dataset, found {len(self.people)}: {self.people}"
            )

        print(f"Loaded dataset with {len(self.people)} people:")
        for person, imgs in self.person_images.items():
            print(f"  - {person}: {len(imgs)} images")

    def __len__(self):
        return sum(len(v) for v in self.person_images.values()) * 3

    def __getitem__(self, idx):
        # Sample anchor person
        anchor_person = random.choice(self.people)

        # Anchor + positive: two different images of the same person
        anchor_path, positive_path = random.sample(self.person_images[anchor_person], 2)

        # Negative: a random image of a different person
        neg_person  = random.choice([p for p in self.people if p != anchor_person])
        neg_path    = random.choice(self.person_images[neg_person])

        anchor_img   = cv2.imread(anchor_path,   0)
        positive_img = cv2.imread(positive_path, 0)
        negative_img = cv2.imread(neg_path,      0)

        if anchor_img is None or positive_img is None or negative_img is None:
            return self.__getitem__(0)  # fallback on corrupt image

        anchor   = preprocess_face(anchor_img,   augment=self.augment)
        positive = preprocess_face(positive_img, augment=self.augment)
        negative = preprocess_face(negative_img, augment=self.augment)

        return anchor, positive, negative


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for A, P, N in val_loader:
            A, P, N = A.to(device), P.to(device), N.to(device)
            total_loss += triplet_loss(model(A), model(P), model(N)).item()
    return total_loss / len(val_loader)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Dataset path: {DATASET_PATH}\n")

    train_ds = TripletDataset(DATASET_PATH, augment=True)
    val_ds   = TripletDataset(DATASET_PATH, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # FIX: pass embedding_dim explicitly so it's always consistent
    model     = FaceEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for A, P, N in train_loader:
            A, P, N = A.to(device), P.to(device), N.to(device)
            optimizer.zero_grad()
            loss = triplet_loss(model(A), model(P), model(N))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val   = validate(model, val_loader, device)

        print(f"Epoch {epoch:3d}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}")

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  ✅ Saved best model  (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE}")
    print("Run build_database.py next to rebuild embeddings.npy.")


if __name__ == "__main__":
    train()