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
print(f"   Found {len(people)} people: {people}")

if len(people) >= 2:
    for i in range(min(3, len(people))):
        for j in range(i+1, min(4, len(people))):
            person1_emb = db[people[i]]['mean']
            person2_emb = db[people[j]]['mean']
            
            similarity_different = cosine_similarity(person1_emb, person2_emb)
            print(f"   Similarity between {people[i]} and {people[j]}: {similarity_different:.3f}")
            
            if similarity_different > 0.8:
                print(f"      ⚠️  WARNING: Very similar embeddings!")
            elif similarity_different > 0.7:
                print(f"      ⚠️  CAUTION: Somewhat similar")
            else:
                print(f"      ✅ Good: Distinct embeddings")

# Test 2: Check embeddings are similar for same person
print("\n2. Checking if same person's embeddings are similar...")
for person in people:
    embeddings = db[person]['embeddings']
    if len(embeddings) >= 2:
        # Check first 5 pairs
        similarities = []
        for i in range(min(5, len(embeddings)-1)):
            sim = cosine_similarity(embeddings[i], embeddings[i+1])
            similarities.append(sim)
        
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        
        print(f"   {person}: avg={avg_sim:.3f}, min={min_sim:.3f}")
        
        if min_sim < 0.5:
            print(f"      ⚠️  WARNING: Some images are very different!")
        elif avg_sim > 0.85:
            print(f"      ✅ Excellent consistency")
        else:
            print(f"      ✅ Good")

# Test 3: Test on actual images from dataset
print("\n3. Testing recognition on actual dataset images...")

# Try to find dataset folder
dataset_paths = ["src/facial_recognition/dataset", "../dataset", "../../dataset"]
dataset_dir = None

for path in dataset_paths:
    if os.path.exists(path):
        dataset_dir = path
        break

if dataset_dir is None:
    print("   ⚠️  Could not find dataset folder!")
    print(f"   Current directory: {os.getcwd()}")
    print("   Skipping image tests...")
else:
    print(f"   Using dataset: {dataset_dir}")
    
    test_count = 0
    correct_count = 0
    
    for person in people:
        person_dir = os.path.join(dataset_dir, person)
        
        if os.path.exists(person_dir):
            images = [f for f in os.listdir(person_dir) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if images:
                # Test first 3 images for each person
                for img_name in images[:3]:
                    test_img_path = os.path.join(person_dir, img_name)
                    img = cv2.imread(test_img_path, 0)
                    
                    if img is not None:
                        test_count += 1
                        
                        # Get embedding
                        try:
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
                            
                            top_match = sorted_scores[0][0]
                            top_score = sorted_scores[0][1]
                            
                            if top_match == person:
                                correct_count += 1
                                status = "✅"
                            else:
                                status = "❌"
                            
                            print(f"   {status} {person}/{img_name}: predicted={top_match} (score={top_score:.3f})")
                            
                            # Show top 3 if wrong
                            if top_match != person:
                                print(f"      Top 3:")
                                for i, (name, score) in enumerate(sorted_scores[:3]):
                                    print(f"        {i+1}. {name}: {score:.3f}")
                        
                        except Exception as e:
                            print(f"   ❌ Error processing {person}/{img_name}: {e}")
    
    if test_count > 0:
        accuracy = (correct_count / test_count) * 100
        print(f"\n   Overall Accuracy: {correct_count}/{test_count} = {accuracy:.1f}%")
        
        if accuracy < 60:
            print("   ⚠️  POOR: Model needs retraining or better data")
        elif accuracy < 80:
            print("   ⚠️  FAIR: Could be improved")
        elif accuracy < 95:
            print("   ✅ GOOD: Model is working well")
        else:
            print("   ✅ EXCELLENT: Model is very accurate")

print("\n" + "="*60)

# Test 4: Simulate live recognition with thresholds
print("\n4. Testing with recognition thresholds...")
threshold = 0.70
min_match_rate = 0.5

if dataset_dir:
    for person in people[:2]:  # Test first 2 people
        person_dir = os.path.join(dataset_dir, person)
        
        if os.path.exists(person_dir):
            images = [f for f in os.listdir(person_dir) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if images:
                test_img_path = os.path.join(person_dir, images[0])
                img = cv2.imread(test_img_path, 0)
                
                if img is not None:
                    with torch.no_grad():
                        face_tensor = preprocess_face(img, augment=False).unsqueeze(0).to(device)
                        emb = model(face_tensor).cpu().numpy()[0]
                    
                    print(f"\n   Testing {person}/{images[0]} with strict thresholds:")
                    
                    for db_person, data in db.items():
                        embeddings = data['embeddings']
                        similarities = []
                        
                        for db_emb in embeddings:
                            sim = cosine_similarity(emb, db_emb)
                            similarities.append(sim)
                        
                        avg_similarity = np.mean(similarities)
                        matches_above_threshold = sum(1 for s in similarities if s > threshold)
                        match_rate = matches_above_threshold / len(similarities)
                        
                        passes = avg_similarity > threshold and match_rate >= min_match_rate
                        status = "✅ PASS" if passes else "❌ FAIL"
                        
                        print(f"      {db_person}: avg={avg_similarity:.3f}, match_rate={match_rate:.2f} {status}")

print("\n" + "="*60)