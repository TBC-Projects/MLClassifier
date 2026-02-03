import cv2
import torch
import numpy as np
from model import FaceEmbeddingNet
from utils import preprocess_face, cosine_similarity
import os

# Load model
device = "cpu"
model = FaceEmbeddingNet().to(device)
model.load_state_dict(torch.load("src/facial_recognition/face_model.pth", map_location=device))
model.eval()

# Load database
db = np.load("src/facial_recognition/embeddings.npy", allow_pickle=True).item()

print("="*60)
print("MODEL DIAGNOSIS")
print("="*60)

# Test 1: Check embeddings are different for different people
print("\n1. Checking if embeddings are distinct between people...")
people = list(db.keys())
if len(people) >= 2:
    person1_emb = db[people[0]]['mean']
    person2_emb = db[people[1]]['mean']
    
    similarity_different = cosine_similarity(person1_emb, person2_emb)
    print(f"   Similarity between {people[0]} and {people[1]}: {similarity_different:.3f}")
    
    if similarity_different > 0.8:
        print("   ⚠️  WARNING: Different people have very similar embeddings!")
        print("   This means the model didn't learn well.")
    else:
        print("   ✅ Good: Different people have distinct embeddings")

# Test 2: Check embeddings are similar for same person
print("\n2. Checking if same person's embeddings are similar...")
for person in people[:2]:  # Test first 2 people
    embeddings = db[person]['embeddings']
    if len(embeddings) >= 2:
        sim = cosine_similarity(embeddings[0], embeddings[1])
        print(f"   {person}: similarity between two images = {sim:.3f}")
        
        if sim < 0.6:
            print(f"   ⚠️  WARNING: Same person has dissimilar embeddings!")
        else:
            print(f"   ✅ Good")

# Test 3: Test on actual images from dataset
print("\n3. Testing recognition on actual dataset images...")
for person in people:
    person_dir = f"dataset/{person}"
    if os.path.exists(person_dir):
        images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if images:
            # Test first image
            test_img_path = os.path.join(person_dir, images[0])
            img = cv2.imread(test_img_path, 0)
            
            if img is not None:
                # Get embedding
                with torch.no_grad():
                    face_tensor = preprocess_face(img, augment=False).unsqueeze(0).to(device)
                    emb = model(face_tensor).cpu().numpy()[0]
                
                # Compare with all people
                scores = {}
                for db_person, data in db.items():
                    sim = cosine_similarity(emb, data['mean'])
                    scores[db_person] = sim
                
                # Sort by score
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n   Testing {person}/{images[0]}:")
                print(f"   Top 3 matches:")
                for i, (name, score) in enumerate(sorted_scores[:3]):
                    marker = "✅" if name == person else "❌"
                    print(f"      {i+1}. {name}: {score:.3f} {marker}")
                
                if sorted_scores[0][0] != person:
                    print(f"   ⚠️  FAILED: Should recognize as {person}, got {sorted_scores[0][0]}")

print("\n" + "="*60)