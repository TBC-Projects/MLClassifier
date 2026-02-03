import cv2
import torch
import numpy as np
from model import FaceEmbeddingNet
from utils import preprocess_face, cosine_similarity
import time
from collections import deque, Counter
import os

class FaceRecognizer:
    def __init__(self, model_path="src/facial_recognition/face_model.pth", db_path="src/facial_recognition/embeddings.npy", 
                 cascade_path="src/facial_recognition/haarcascade_frontalface_default.xml"):
        
        # Get script directory for absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Make paths absolute
        if not os.path.isabs(model_path):
            model_path = os.path.join(script_dir, model_path)
        if not os.path.isabs(db_path):
            db_path = os.path.join(script_dir, db_path)
        if not os.path.isabs(cascade_path):
            cascade_path = os.path.join(script_dir, cascade_path)
        
        # Load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model = FaceEmbeddingNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from: {model_path}")
        
        # Load database
        self.db = np.load(db_path, allow_pickle=True).item()
        print(f"Database loaded from: {db_path}")
        
        # Face detector
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise ValueError(f"Could not load cascade classifier from {cascade_path}")
        print(f"Face detector loaded from: {cascade_path}")
        
        # Recognition parameters
        self.threshold = 0.70  # Stricter threshold - increase to 0.75 or 0.80 for even fewer false positives
        self.min_matches = 0.5  # At least 50% of embeddings must match
        
        # Temporal smoothing parameters
        self.recognition_history = {}  # Stores recent recognitions per face
        self.history_length = 7  # Use last 7 frames for voting
        self.min_agreement = 0.6  # Require 60% agreement across frames
        
        # Display settings
        self.show_fps = True
        self.show_confidence = True
        
        # Performance tracking
        self.frame_times = []
        
        print(f"\n{'='*50}")
        print(f"Face Recognition System Initialized")
        print(f"{'='*50}")
        print(f"People in database: {len(self.db)}")
        for person in self.db.keys():
            num_embeddings = len(self.db[person]['embeddings'])
            print(f"  - {person}: {num_embeddings} reference images")
        print(f"\nRecognition Settings:")
        print(f"  - Similarity threshold: {self.threshold}")
        print(f"  - Minimum match rate: {self.min_matches}")
        print(f"  - Temporal smoothing: {self.history_length} frames")
        print(f"  - Agreement threshold: {self.min_agreement}")
        print(f"{'='*50}\n")
    
    def recognize_face(self, face_img):
    
    # Get embedding for current face
        with torch.no_grad():
            face_tensor = preprocess_face(face_img, augment=False).unsqueeze(0).to(self.device)
            emb = self.model(face_tensor).cpu().numpy()[0]
        
        # Compare with each person in database
        scores_list = []
        
        for person, data in self.db.items():
            # Compare with all embeddings for this person, not just mean
            embeddings = data['embeddings']
            similarities = []
            
            for db_emb in embeddings:
                sim = cosine_similarity(emb, db_emb)
                similarities.append(sim)
            
            # Calculate statistics
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            matches_above_threshold = sum(1 for s in similarities if s > self.threshold)
            match_rate = matches_above_threshold / len(similarities)
            
            scores_list.append({
                'person': person,
                'avg_sim': avg_similarity,
                'max_sim': max_similarity,
                'match_rate': match_rate
            })
        
        # Sort by average similarity
        scores_list.sort(key=lambda x: x['avg_sim'], reverse=True)
        
        # Get best and second-best
        best = scores_list[0] if len(scores_list) > 0 else None
        second_best = scores_list[1] if len(scores_list) > 1 else None
        
        if best is None:
            return "Unknown", 0
        
        # Calculate margin between best and second-best
        if second_best:
            score_margin = best['avg_sim'] - second_best['avg_sim']
        else:
            score_margin = 1.0  # Only one person in database
        
        # Strict rejection criteria - ALL must pass:
        # 1. Score above threshold
        # 2. Enough matches above threshold
        # 3. Clear margin from second place (important for similar people!)
        if (best['avg_sim'] < self.threshold or 
            best['match_rate'] < self.min_matches or
            score_margin < 0.20):  # Require 0.20 gap for similar people
            
            return "Unknown", 0
        
        return best['person'], best['avg_sim']
    
    def smooth_recognition(self, face_id, name, score):
        """
        Apply temporal smoothing to reduce flickering between recognitions.
        Uses voting across recent frames.
        
        Args:
            face_id: Unique identifier for face position
            name: Recognized name for current frame
            score: Confidence score for current frame
            
        Returns:
            tuple: (stable_name, stable_score)
        """
        # Initialize history for new faces
        if face_id not in self.recognition_history:
            self.recognition_history[face_id] = deque(maxlen=self.history_length)
        
        # Add current recognition to history
        self.recognition_history[face_id].append((name, score))
        
        # Vote: find most common name in recent history
        names = [n for n, s in self.recognition_history[face_id]]
        name_counts = Counter(names)
        most_common_name, count = name_counts.most_common(1)[0]
        
        # Require minimum agreement threshold
        agreement_rate = count / len(names)
        
        if agreement_rate >= self.min_agreement:
            # Calculate average score for the winning name
            scores = [s for n, s in self.recognition_history[face_id] if n == most_common_name]
            avg_score = np.mean(scores) if scores else 0
            return most_common_name, avg_score
        else:
            # Not enough agreement - return Unknown
            return "Unknown", 0.0
    
    def cleanup_old_faces(self, current_face_ids):
        """Remove history for faces no longer in frame"""
        face_ids_to_remove = [fid for fid in self.recognition_history.keys() 
                               if fid not in current_face_ids]
        for fid in face_ids_to_remove:
            del self.recognition_history[fid]
    
    def run(self, camera_id=0):
        """
        Run live face recognition on camera feed.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
        """
        cap = cv2.VideoCapture(camera_id)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        print("Starting face recognition...")
        print("Press ESC to quit\n")
        
        frame_count = 0
        
        try:
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
                    minSize=(60, 60),  # Minimum face size
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Track current face IDs for cleanup
                current_face_ids = []
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Recognize face
                    name, score = self.recognize_face(face_roi)
                    
                    # Create face ID based on grid position (for temporal smoothing)
                    face_id = f"{x//50}_{y//50}"
                    current_face_ids.append(face_id)
                    
                    # Apply temporal smoothing
                    stable_name, stable_score = self.smooth_recognition(face_id, name, score)
                    
                    # Choose color based on recognition
                    if stable_name != "Unknown":
                        color = (0, 255, 0)  # Green for recognized
                        thickness = 2
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        thickness = 2
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                    
                    # Create label
                    if stable_name != "Unknown" and self.show_confidence:
                        label = f"{stable_name} ({stable_score:.2f})"
                    else:
                        label = stable_name
                    
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(frame, (x, y-label_h-10), (x+label_w, y), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Cleanup old faces no longer in frame
                self.cleanup_old_faces(current_face_ids)
                
                # Calculate and display FPS
                if self.show_fps:
                    elapsed = time.time() - start_time
                    fps = 1 / elapsed if elapsed > 0 else 0
                    self.frame_times.append(fps)
                    
                    # Keep only last 30 frames for average
                    if len(self.frame_times) > 30:
                        self.frame_times.pop(0)
                    
                    avg_fps = np.mean(self.frame_times)
                    
                    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display face count
                cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Face Recognition", frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("\nQuitting...")
                    break
                elif key == ord('s'):  # 's' to save screenshot
                    screenshot_name = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_name, frame)
                    print(f"Screenshot saved: {screenshot_name}")
                
                frame_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nProcessed {frame_count} frames")
            if self.frame_times:
                print(f"Average FPS: {np.mean(self.frame_times):.1f}")

if __name__ == "__main__":
    # Initialize and run face recognizer
    try:
        recognizer = FaceRecognizer(
            model_path="face_model.pth",
            db_path="embeddings.npy",
            cascade_path="haarcascade_frontalface_default.xml"
        )
        recognizer.run(camera_id=0)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()