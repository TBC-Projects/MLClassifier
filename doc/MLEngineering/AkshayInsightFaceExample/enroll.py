import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

DB_PATH = "face_db"
os.makedirs(DB_PATH, exist_ok=True)

app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(320, 320))

cap = cv2.VideoCapture(0)

name = input("Enter person's name: ").strip()

embeddings = []
print("Press SPACE to capture face | ESC to finish")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("Enroll Face", frame)
    key = cv2.waitKey(1)

    if key == 32 and len(faces) == 1:  # SPACE
        emb = faces[0].embedding
        emb /= np.linalg.norm(emb)
        embeddings.append(emb)
        print(f"Captured {len(embeddings)}")

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

embeddings = np.array(embeddings)

# Save
if os.path.exists(f"{DB_PATH}/embeddings.npy"):
    old_emb = np.load(f"{DB_PATH}/embeddings.npy")
    old_labels = np.load(f"{DB_PATH}/labels.npy", allow_pickle=True)

    embeddings = np.vstack([old_emb, embeddings])
    labels = np.hstack([old_labels, [name]*len(embeddings)])
else:
    labels = np.array([name]*len(embeddings))

np.save(f"{DB_PATH}/embeddings.npy", embeddings)
np.save(f"{DB_PATH}/labels.npy", labels)

print("Enrollment complete.")
