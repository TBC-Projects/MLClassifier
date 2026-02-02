# build_database.py - Improved
import os
import cv2
import torch
import numpy as np
from model import FaceEmbeddingNet
from utils import preprocess_face

def build_database():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FaceEmbeddingNet().to(device)
    model.load_state_dict(torch.load("face_model.pth", map_location=device))
    model.eval()
    
    db = {}
    
    with torch.no_grad():
        for person in os.listdir("dataset"):
            person_path = os.path.join("dataset", person)
            if not os.path.isdir(person_path):
                continue
            
            embs = []
            for img_name in os.listdir(person_path):
                if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, 0)
                
                if img is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                
                # Get embedding
                face_tensor = preprocess_face(img, augment=False).unsqueeze(0).to(device)
                emb = model(face_tensor).cpu().numpy()[0]
                embs.append(emb)
            
            if len(embs) > 0:
                # Store all embeddings, not just mean
                db[person] = {
                    'embeddings': np.array(embs),
                    'mean': np.mean(embs, axis=0)
                }
                print(f"Added {person}: {len(embs)} embeddings")
            else:
                print(f"Warning: No valid images for {person}")
    
    np.save("embeddings.npy", db, allow_pickle=True)
    print(f"\nDatabase built with {len(db)} people")

if __name__ == "__main__":
    build_database()