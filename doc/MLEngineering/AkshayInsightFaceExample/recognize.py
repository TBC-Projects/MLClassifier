import cv2
import numpy as np
from insightface.app import FaceAnalysis
from utils import identify, normalize

DB_PATH = "face_db"

db_embeddings = normalize(np.load(f"{DB_PATH}/embeddings.npy"))
labels = np.load(f"{DB_PATH}/labels.npy", allow_pickle=True)

app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(320, 320))

cap = cv2.VideoCapture(0)

FRAME_SKIP = 5
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    faces = app.get(frame)

    for face in faces:
        emb = face.embedding
        emb /= np.linalg.norm(emb)

        name, score = identify(emb, db_embeddings, labels)

        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{name} ({score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
