import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import FaceEmbeddingNet
from utils import preprocess_face, augment_face
from sklearn.model_selection import train_test_split

# Get absolute path to dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")

class TripletDataset(Dataset):
    def __init__(self, root, augment=True):
        self.data = {}
        self.augment = augment
        
        # Check if dataset exists
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        
        for person in os.listdir(root):
            p = os.path.join(root, person)
            if os.path.isdir(p):
                imgs = [os.path.join(p, f) for f in os.listdir(p) 
                       if f.endswith(('.jpg', '.png', '.jpeg'))]
                if len(imgs) >= 2:
                    self.data[person] = imgs
        
        self.people = list(self.data.keys())
        if len(self.people) < 2:
            raise ValueError(f"Need at least 2 people in dataset. Found: {len(self.people)}")
        
        print(f"Loaded {len(self.people)} people: {self.people}")
        for person, imgs in self.data.items():
            print(f"  {person}: {len(imgs)} images")
    
    def __len__(self):
        return len(self.people) * 100
    
    def __getitem__(self, idx):
        anchor_person = random.choice(self.people)
        anchor_img, positive_img = random.sample(self.data[anchor_person], 2)
        negative_person = random.choice([p for p in self.people if p != anchor_person])
        negative_img = random.choice(self.data[negative_person])
        
        anchor = self._load_img(anchor_img)
        positive = self._load_img(positive_img)
        negative = self._load_img(negative_img)
        
        return anchor, positive, negative
    
    def _load_img(self, path):
        img = cv2.imread(path, 0)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return preprocess_face(img, augment=self.augment)

def triplet_loss(anchor, positive, negative, margin=0.5):
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for A, P, N in val_loader:
            A, P, N = A.to(device), P.to(device), N.to(device)
            loss = triplet_loss(model(A), model(P), model(N))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Looking for dataset at: {DATASET_PATH}")
    
    dataset = TripletDataset(DATASET_PATH, augment=True)
    val_dataset = TripletDataset(DATASET_PATH, augment=False)
    
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = FaceEmbeddingNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3  # ← Removed verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(5):
        model.train()
        train_loss = 0
        
        for batch_idx, (A, P, N) in enumerate(train_loader):
            A, P, N = A.to(device), P.to(device), N.to(device)
            
            optimizer.zero_grad()
            loss = triplet_loss(model(A), model(P), model(N))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        val_loss = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/5 - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(SCRIPT_DIR, "face_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print("Early stopping triggered")
                break
    
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()