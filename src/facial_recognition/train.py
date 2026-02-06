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


def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Stricter margin forces the model to create more distinct embeddings.
    This helps distinguish similar-looking people.
    """
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()


class TripletDataset(Dataset):
    """
    ADDED: Dataset class for generating triplets (anchor, positive, negative)
    """
    def __init__(self, dataset_path, augment=False):
        self.dataset_path = dataset_path
        self.augment = augment
        self.data = []
        
        # Load all images organized by person
        self.person_images = {}
        
        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path):
                continue
            
            images = []
            for img_name in os.listdir(person_path):
                if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                img_path = os.path.join(person_path, img_name)
                images.append(img_path)
            
            if len(images) >= 2:  # Need at least 2 images for anchor and positive
                self.person_images[person] = images
        
        self.people = list(self.person_images.keys())
        
        if len(self.people) < 2:
            raise ValueError(f"Need at least 2 people in dataset, found {len(self.people)}")
        
        print(f"Loaded dataset with {len(self.people)} people:")
        for person, images in self.person_images.items():
            print(f"  - {person}: {len(images)} images")
    
    def __len__(self):
        # Generate multiple triplets per epoch
        total_images = sum(len(imgs) for imgs in self.person_images.values())
        return total_images * 3  # 3x multiplier for variety
    
    def __getitem__(self, idx):
        # Select anchor person randomly
        anchor_person = random.choice(self.people)
        
        # Select two different images from same person (anchor and positive)
        anchor_img_path, positive_img_path = random.sample(
            self.person_images[anchor_person], 2
        )
        
        # Select negative person (different from anchor)
        negative_person = random.choice([p for p in self.people if p != anchor_person])
        negative_img_path = random.choice(self.person_images[negative_person])
        
        # Load images
        anchor_img = cv2.imread(anchor_img_path, 0)
        positive_img = cv2.imread(positive_img_path, 0)
        negative_img = cv2.imread(negative_img_path, 0)
        
        # Check if images loaded successfully
        if anchor_img is None or positive_img is None or negative_img is None:
            # Fallback to first valid triplet
            return self.__getitem__(0)
        
        # Preprocess with optional augmentation
        anchor_tensor = preprocess_face(anchor_img, augment=self.augment)
        positive_tensor = preprocess_face(positive_img, augment=self.augment)
        negative_tensor = preprocess_face(negative_img, augment=self.augment)
        
        return anchor_tensor, positive_tensor, negative_tensor


def validate(model, val_loader, device):
    """
    ADDED: Validation function to compute validation loss
    """
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for A, P, N in val_loader:
            A, P, N = A.to(device), P.to(device), N.to(device)
            loss = triplet_loss(model(A), model(P), model(N))
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Looking for dataset at: {DATASET_PATH}")
    
    dataset = TripletDataset(DATASET_PATH, augment=True)
    val_dataset = TripletDataset(DATASET_PATH, augment=False)
    
    train_loader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=0)
    
    model = FaceEmbeddingNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(100):
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
        
        print(f"Epoch {epoch+1}/100 - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
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
            if patience_counter >= 15:
                print("Early stopping triggered")
                break
    
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()