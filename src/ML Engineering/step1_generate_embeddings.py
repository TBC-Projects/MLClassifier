# DO NOT RUN, NOT CONFIRMED TO WORK
# step1_generate_embeddings.py
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
import cv2, torch, os, numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import Image

# Initialize MTCNN (face detector)
mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

# Initialize ArcFace embedding model
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Directory of known people
KNOWN_DIR = "known_faces"
embeddings = []
labels = []

for person_name in os.listdir(KNOWN_DIR):
    person_dir = os.path.join(KNOWN_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Detect face using MTCNN
        boxes, probs = mtcnn.detect(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if boxes is None:
            print(f"No face found in {img_name}")
            continue

        # Extract embeddings with ArcFace
        faces = app.get(img)
        if len(faces) == 0:
            continue

        emb = faces[0].embedding  # 512-d embedding
        embeddings.append(emb)
        labels.append(person_name)

# Convert to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Encode labels to integers
le = LabelEncoder()
y = le.fit_transform(labels)

# Save embeddings and label encoder
with open("face_embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, y, le), f)

print("✅ Saved embeddings and label encoder.")
