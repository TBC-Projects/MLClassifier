# DO NOT RUN, NOT CONFIRMED TO WORK
# step3_live_recognition.py
import cv2, torch, pickle, numpy as np
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
from PIL import Image

# Load models
mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load classifier + label encoder
with open("face_classifier.pkl", "rb") as f:
    clf, le = pickle.load(f)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(Image.fromarray(rgb_frame))

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(x) for x in box]
            face_img = frame[y1:y2, x1:x2]

            faces = app.get(face_img)
            if len(faces) == 0:
                continue

            emb = faces[0].embedding.reshape(1, -1)

            # Predict using classifier
            pred = clf.predict(emb)
            name = le.inverse_transform(pred)[0]

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
