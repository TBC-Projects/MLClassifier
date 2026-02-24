"""
Live Facial Recognition Pipeline for NVIDIA Jetson Nano
Using MobileFaceNet and InsightFace
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import pickle
import os
from pathlib import Path
import time
import csv
from datetime import datetime
from pathlib import Path
import serial

# Works when run as: python attendance_logger.py
CSV_FILE = Path(__file__).parent / "attendance.csv"
last_logged = set()

with open("/Users/hudsonwong/Work:School/College/Boring_Club/MLClassifier/src/facenet_model/attendance.csv", "r", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1] not in last_logged:
            last_logged.add(row[1])
#
#ser = serial.Serial(
    #port='/dev/cu.usbserial-0001',  # Change this
    #baudrate=115200,
    #timeout=1
#)

time.sleep(2)  # Wait for device (important for Arduino/ESP)

class FaceRecognitionPipeline:
    def __init__(self, database_path='face_database', model_name='buffalo_sc'):
        """
        Initialize the face recognition pipeline
        
        Args:
            database_path: Path to store face embeddings database
            model_name: InsightFace model name ('buffalo_sc' is lightweight)
        """
        self.database_path = Path(database_path)
        self.database_path.mkdir(exist_ok=True)
        
        # Initialize InsightFace with lightweight model
        print("Loading face detection and recognition models...")
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        # det_size=(320, 320) for faster detection on Jetson Nano
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        
        # Load or initialize face database
        self.face_database = {}
        self.database_file = self.database_path / 'embeddings.pkl'
        self.load_database()
        
        # Recognition threshold (lower = stricter)
        self.threshold = 0.6
        
    def extract_embedding(self, image):
        """
        Extract face embedding from image
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            embedding vector or None if no face detected
        """
        faces = self.app.get(image)
        
        if len(faces) == 0:
            return None
        
        # Return embedding of the largest face
        largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return largest_face.embedding
    
    def add_person_to_database(self, person_name, image_folder):
        """
        Add a person to the database using multiple images
        
        Args:
            person_name: Name/ID of the person
            image_folder: Path to folder containing person's images
        """
        print(f"Processing images for {person_name}...")
        image_folder = Path(image_folder)
        embeddings = []
        
        # Supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for img_path in image_folder.iterdir():
            if img_path.suffix.lower() in image_extensions:
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                embedding = self.extract_embedding(image)
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    print(f"Warning: No face detected in {img_path}")
        
        if len(embeddings) == 0:
            print(f"Error: No valid embeddings extracted for {person_name}")
            return False
        
        # Store average embedding for the person
        avg_embedding = np.mean(embeddings, axis=0)
        self.face_database[person_name] = avg_embedding
        
        print(f"Added {person_name} with {len(embeddings)} embeddings")
        return True
    
    def save_database(self):
        """Save face database to disk"""
        with open(self.database_file, 'wb') as f:
            pickle.dump(self.face_database, f)
        print(f"Database saved to {self.database_file}")
    
    def load_database(self):
        """Load face database from disk"""
        if self.database_file.exists():
            with open(self.database_file, 'rb') as f:
                self.face_database = pickle.load(f)
            print(f"Loaded database with {len(self.face_database)} people")
        else:
            print("No existing database found. Starting fresh.")
    
    def recognize_face(self, embedding):
        """
        Recognize a face by comparing embedding to database
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            (person_name, confidence) or (None, 0) if no match
        """
        if len(self.face_database) == 0:
            return None, 0
        
        best_match = None
        best_distance = float('inf')
        
        for person_name, stored_embedding in self.face_database.items():
            # Calculate cosine distance
            distance = cosine(embedding, stored_embedding)
            
            if distance < best_distance:
                best_distance = distance
                best_match = person_name
        
        # Convert distance to similarity score
        similarity = 1 - best_distance
        
        if similarity >= self.threshold:
            return best_match, similarity
        else:
            return "Unknown", similarity
    
    def run_live_recognition(self, camera_id=0, display=True):
        """
        Run live face recognition from camera
        
        Args:
            camera_id: Camera device ID (0 for default)
            display: Whether to display video feed
        """
        cap = cv2.VideoCapture(camera_id)
        
        # Set camera resolution (lower resolution = faster processing)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting live recognition. Press 'q' to quit.")
        
        # For FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        # Process every N frames to improve speed
        frame_skip = 2
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Skip frames for better performance
            if frame_count % frame_skip != 0:
                if display:
                    cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Detect and recognize faces
            faces = self.app.get(frame)
            
            for face in faces:
                # Get bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Recognize face
                person_name, confidence = self.recognize_face(face.embedding)
                if confidence > 0.6:
                    if should_log(person_name):
                        log_attendance(person_name)
                
                # Draw bounding box
                color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{person_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 10:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display FPS
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if display:
                cv2.imshow('Face Recognition', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def should_log(name):
    now = datetime.now()
    if name in last_logged:
        return False
    return True

def log_attendance(name):
    """Log attendance with timestamp."""
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name])
    
    print(f"✅ Logged: {name}")
    print(f"📁 Saved to: {CSV_FILE.absolute()}")
    last_logged.add(name)
    #ser.write(b'complete')


def main():
    """Example usage of the pipeline"""
    
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline()
    
    # Example: Build database from folders
    # Assuming you have a structure like:
    # training_data/
    #   ├── person1/
    #   │   ├── img1.jpg
    #   │   ├── img2.jpg
    #   │   └── ...
    #   ├── person2/
    #   │   └── ...
    
    build_database = input("Do you want to build/rebuild the database? (y/n): ")
    
    if build_database.lower() == 'y':
        training_data_path = input("Enter path to training data folder: ")
        training_data_path = Path(training_data_path)
        
        if training_data_path.exists():
            for person_folder in training_data_path.iterdir():
                if person_folder.is_dir():
                    person_name = person_folder.name
                    pipeline.add_person_to_database(person_name, person_folder)
            
            pipeline.save_database()
        else:
            print(f"Error: Path {training_data_path} does not exist")
            return
    
    # Run live recognition
    print("\nStarting live face recognition...")
    pipeline.run_live_recognition(camera_id=0, display=True)


if __name__ == "__main__":
    main()
