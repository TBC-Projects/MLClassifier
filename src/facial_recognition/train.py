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

# At the top, update the triplet loss function:

def triplet_loss(anchor, positive, negative, margin=0.2):  # Reduced from 0.5 to 0.2
    """
    Stricter margin forces the model to create more distinct embeddings.
    This helps distinguish similar-looking people.
    """
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()

# In the train() function, update:

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Looking for dataset at: {DATASET_PATH}")
    
    dataset = TripletDataset(DATASET_PATH, augment=True)
    val_dataset = TripletDataset(DATASET_PATH, augment=False)
    
    train_loader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=0)  # Reduced batch size for deeper model
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=0)
    
    model = FaceEmbeddingNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5  # Increased patience
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(100):  # Increased from 50 to 100
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
            if patience_counter >= 15:  # Increased from 7 to 15
                print("Early stopping triggered")
                break
    
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")