import cv2
import mediapipe as mp
import numpy as np
import argparse
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceExtractor:
    def __init__(self, model_path="face_landmarker.task", num_faces=3):
        # Initialize MediaPipe Face Landmarker using Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=num_faces,  # Allow up to 3 faces
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.num_faces = num_faces
        
        # Drawing utilities - using OpenCV for drawing since MediaPipe solutions API is deprecated
        self.mp_drawing = None  # We'll use OpenCV for drawing
        
    def extract_from_image(self, image_path, output_path=None):
        """
        Extract face region and feature points from an image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save output image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect face landmarks
        detection_result = self.face_landmarker.detect(mp_image)
        
        # Create a copy for drawing
        annotated_image = image.copy()
        
        # Draw face detection bounding boxes and landmarks for each face
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
        if detection_result.face_landmarks:
            print(f"Found {len(detection_result.face_landmarks)} face(s)")
            
            for face_idx, face_landmarks in enumerate(detection_result.face_landmarks):
                color = colors[face_idx % len(colors)]
                print(f"\n--- Face {face_idx + 1} ---")
                
                # Get bounding box from landmarks
                xs = [landmark.x * w for landmark in face_landmarks]
                ys = [landmark.y * h for landmark in face_landmarks]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Draw face number label
                cv2.putText(annotated_image, f"Face {face_idx + 1}", (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw connections for face contours (approximate)
                # Key landmarks indices for face outline
                key_points = [
                    (10, 151), (151, 9), (9, 10),  # Forehead
                    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),  # Left side
                    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362),  # Right side
                ]
                
                # Draw some key connections
                for start_idx, end_idx in [(10, 151), (33, 7), (263, 249), (10, 9), (151, 9)]:
                    if start_idx < len(face_landmarks) and end_idx < len(face_landmarks):
                        pt1 = (int(face_landmarks[start_idx].x * w), int(face_landmarks[start_idx].y * h))
                        pt2 = (int(face_landmarks[end_idx].x * w), int(face_landmarks[end_idx].y * h))
                        cv2.line(annotated_image, pt1, pt2, color, 1)
                
                # Print some key landmark coordinates
                print("Key landmarks (normalized coordinates 0-1):")
                landmark_names = {
                    10: "Forehead center",
                    33: "Left eye outer corner",
                    263: "Right eye outer corner",
                    1: "Nose tip",
                    13: "Upper lip",
                    14: "Lower lip",
                    152: "Chin",
                }
                
                for idx, name in landmark_names.items():
                    if idx < len(face_landmarks):
                        landmark = face_landmarks[idx]
                        pixel_x = int(landmark.x * w)
                        pixel_y = int(landmark.y * h)
                        print(f"  {name}: ({landmark.x:.3f}, {landmark.y:.3f}) -> Pixel: ({pixel_x}, {pixel_y})")
        
        # Save or display
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"\nOutput saved to: {output_path}")
        else:
            # Display the image
            cv2.imshow('Face Detection and Landmarks', annotated_image)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated_image, detection_result
    
    def extract_from_webcam(self):
        """
        Extract face region and feature points from webcam feed.
        Press 'q' to quit.
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        print("=" * 60)
        print("Webcam started!")
        print("TO STOP: Press 'q' or 'ESC' key, or close the window")
        print("=" * 60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect face landmarks
            detection_result = self.face_landmarker.detect(mp_image)
            
            # Draw landmarks for each detected face with different colors
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
            if detection_result.face_landmarks:
                for idx, face_landmarks in enumerate(detection_result.face_landmarks):
                    # Use different color for each face (cycle through colors if more than 3)
                    color = colors[idx % len(colors)]
                    
                    # Draw bounding box
                    xs = [landmark.x * w for landmark in face_landmarks]
                    ys = [landmark.y * h for landmark in face_landmarks]
                    x_min, x_max = int(min(xs)), int(max(xs))
                    y_min, y_max = int(min(ys)), int(max(ys))
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # Draw face number label
                    cv2.putText(frame, f"Face {idx + 1}", (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add text overlay showing how to quit and face count
            cv2.putText(frame, "Press 'q' or 'ESC' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces detected: {len(detection_result.face_landmarks)}/{self.num_faces}", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Detection and Landmarks', frame)
            
            # Check for 'q' or ESC key (27 is ESC key code)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nStopping webcam...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def capture_face_dataset(self, person_folder="person1", num_images=20, image_size=(160, 160)):
        """
        Capture face images from webcam and save them to a folder.
        Takes specified number of images when face is detected.
        
        Args:
            person_folder: Folder name to save images (default: "person1")
            num_images: Number of images to capture (default: 20)
            image_size: Size of saved images as (width, height) (default: (160, 160))
        """
        import os
        import time
        
        # Create the folder if it doesn't exist
        folder_path = Path(person_folder)
        folder_path.mkdir(exist_ok=True)
        print(f"Images will be saved to: {folder_path.absolute()}")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        images_captured = 0
        last_capture_time = 0
        capture_interval = 0.5  # Capture one image every 0.5 seconds
        
        print("=" * 60)
        print(f"Face Dataset Capture Mode")
        print(f"Target: {num_images} images")
        print(f"Image size: {image_size[0]}x{image_size[1]} pixels")
        print(f"Press 'q' or 'ESC' to stop")
        print("=" * 60)
        print("\nPosition yourself in front of the camera...")
        print("Starting in 5 seconds...\n")
        
        # 5-second countdown before starting
        countdown_start = time.time()
        countdown_duration = 5.0
        
        while images_captured < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect face landmarks
            detection_result = self.face_landmarker.detect(mp_image)
            
            current_time = time.time()
            elapsed_time = current_time - countdown_start
            remaining_countdown = max(0, countdown_duration - elapsed_time)
            
            # Draw landmarks and capture if face detected (after countdown)
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
            face_detected = False
            can_capture = elapsed_time >= countdown_duration
            
            if detection_result.face_landmarks:
                # Use the first detected face
                face_landmarks = detection_result.face_landmarks[0]
                face_detected = True
                color = colors[0]
                
                # Get bounding box from landmarks
                xs = [landmark.x * w for landmark in face_landmarks]
                ys = [landmark.y * h for landmark in face_landmarks]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                
                # Add padding to ensure we capture the full face
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Draw bounding box only (no landmarks)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Capture image if enough time has passed and countdown is finished
                if can_capture and current_time - last_capture_time >= capture_interval:
                    # Crop face region
                    face_roi = frame[y_min:y_max, x_min:x_max]
                    
                    if face_roi.size > 0:
                        # Resize to target size
                        face_resized = cv2.resize(face_roi, image_size, interpolation=cv2.INTER_AREA)
                        
                        # Save the image
                        image_filename = folder_path / f"picture{images_captured + 1}.jpg"
                        cv2.imwrite(str(image_filename), face_resized)
                        
                        images_captured += 1
                        last_capture_time = current_time
                        
                        print(f"Captured image {images_captured}/{num_images}: {image_filename.name}")
            
            # Display status on frame
            if remaining_countdown > 0:
                status_text = f"Starting in {int(remaining_countdown) + 1} seconds..."
                status_color = (0, 165, 255)  # Orange
            else:
                status_text = f"Captured: {images_captured}/{num_images}"
                if face_detected:
                    status_text += " [Face Detected - Capturing...]"
                else:
                    status_text += " [No Face Detected]"
                status_color = (0, 255, 0)  # Green
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, "Press 'q' or 'ESC' to stop", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Dataset Capture', frame)
            
            # Check for 'q' or ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print(f"\nStopped early. Captured {images_captured}/{num_images} images.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if images_captured == num_images:
            print(f"\n{'=' * 60}")
            print(f"SUCCESS! Captured all {num_images} images.")
            print(f"Images saved to: {folder_path.absolute()}")
            print(f"{'=' * 60}")
        else:
            print(f"\nCaptured {images_captured} out of {num_images} images.")
            print(f"Images saved to: {folder_path.absolute()}")

def main():
    parser = argparse.ArgumentParser(description='Extract face region and feature points using MediaPipe')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to save output image (optional)')
    parser.add_argument('--webcam', action='store_true', help='Use webcam instead of image')
    parser.add_argument('--capture', action='store_true', help='Capture face dataset (20 images, 160x160)')
    parser.add_argument('--person', type=str, default='person1', help='Person folder name for dataset capture (default: person1)')
    parser.add_argument('--num-images', type=int, default=20, help='Number of images to capture (default: 20)')
    
    args = parser.parse_args()
    
    extractor = FaceExtractor()
    
    if args.capture:
        extractor.capture_face_dataset(person_folder=args.person, num_images=args.num_images)
    elif args.webcam:
        extractor.extract_from_webcam()
    elif args.image:
        extractor.extract_from_image(args.image, args.output)
    else:
        # Try to use a sample image if available, or prompt user
        print("Usage:")
        print("  python face_extraction.py --image <image_path> [--output <output_path>]")
        print("  python face_extraction.py --webcam")
        print("  python face_extraction.py --capture [--person <folder_name>] [--num-images <number>]")
        print("\nExample:")
        print("  python face_extraction.py --image photo.jpg --output result.jpg")
        print("  python face_extraction.py --capture --person person1 --num-images 20")

if __name__ == "__main__":
    main()
