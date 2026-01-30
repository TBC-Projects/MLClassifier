# recognize_live.py - Jetson optimized
import cv2
import torch
import numpy as np
from model import FaceEmbeddingNet
from utils import preprocess_face, cosine_similarity
import time

class FaceRecognizer:
    def __init__(self, model_path="face_model.pth", db_path="embeddings.npy", 
                 cascade_path="haarcascade_frontalface_default.xml"):
        
        # Load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FaceEmbeddingNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load database
        self.db = np.load(db_path, allow_pickle=True).item()
        
        # Face detector
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise ValueError(f"Could not load cascade classifier from {cascade_path}")
        
        # Recognition threshold
        self.threshold = 0.6
        
        # Performance tracking
        self.frame_times = []
    
    def recognize_face(self, face_img):
        """Get embedding and match against database"""
        with torch.no_grad():
            face_tensor = preprocess_face(face_img, augment=False).to(self.device)
            emb = self.model(face_tensor).cpu().numpy()[0]
        
        best_match = "Unknown"
        best_score = 0
        
        for person, data in self.db.items():
            # Compare with mean embedding
            score = cosine_similarity(emb, data['mean'])
            
            if score > best_score:
                best_match = person
                best_score = score
        
        if best_score < self.threshold:
            best_match = "Unknown"
        
        return best_match, best_score
    
    def run(self, camera_id=0):
        """Run live recognition"""
        cap = cv2.VideoCapture(camera_id)
        
        # Set camera properties for Jetson
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting face recognition... Press ESC to quit")
        
        frame_count = 0
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                
                # Recognize
                name, score = self.recognize_face(face_roi)
                
                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label
                label = f"{name} ({score:.2f})"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1 / elapsed if elapsed > 0 else 0
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow("Face Recognition", frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.run()