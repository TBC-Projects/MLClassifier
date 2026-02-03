import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import urllib.request
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def download_face_landmarker_model(model_path="face_landmarker.task"):
    """
    Download the MediaPipe face landmarker model if it doesn't exist.
    
    Args:
        model_path: Path where the model should be saved
    """
    model_file = Path(model_path)
    
    if model_file.exists():
        print(f"Model file found at: {model_path}")
        return model_path
    
    print(f"Model file not found. Downloading face_landmarker.task...")
    print("This may take a few moments...")
    
    # Official MediaPipe model URL
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker.task/float16/1/face_landmarker.task"
    
    try:
        # Download the model
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model downloaded successfully to: {model_path}")
        return model_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model file. Error: {e}\n"
            f"Please manually download the model from:\n"
            f"{model_url}\n"
            f"And save it as: {model_path}"
        )

class FaceExtractor:
    def __init__(self, model_path="face_landmarker.task", num_faces=3):
        # Download model if it doesn't exist
        model_path = download_face_landmarker_model(model_path)
        
        # Initialize MediaPipe Face Landmarker using Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,  # Enable for better face alignment
            num_faces=num_faces,  # Allow up to 3 faces
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.num_faces = num_faces
        
        # Drawing utilities - using OpenCV for drawing since MediaPipe solutions API is deprecated
        self.mp_drawing = None  # We'll use OpenCV for drawing
    
    def _open_camera(self):
        """
        Helper function to open webcam with better error handling.
        Tries multiple camera indices and provides helpful error messages.
        
        Returns:
            cv2.VideoCapture object or None if failed
        """
        # Try to open webcam - try multiple indices on macOS
        cap = None
        for camera_idx in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Successfully opened camera {camera_idx}")
                    return cap
                else:
                    cap.release()
                    cap = None
            else:
                if cap:
                    cap.release()
                cap = None
        
        # If we get here, no camera worked
        print("ERROR: Could not open webcam!")
        return None
        
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
                
                # Draw all landmarks as small circles
                for landmark in face_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(annotated_image, (x, y), 1, color, -1)
                
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
        cap = self._open_camera()
        
        if cap is None:
            raise ValueError("Could not open webcam. Please check camera permissions.")
        
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
                    
                    # Draw landmarks as small circles
                    for landmark in face_landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 1, color, -1)
            
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
    
    def capture_aligned_faces(self, num_captures=20, output_folder="photosML", crop_size=160, person_folder=None):
        """
        Capture aligned face images from webcam at multiple timestamps.
        
        Args:
            num_captures: Number of face images to capture (default: 20)
            output_folder: Folder to save captured images when person_folder is not set (default: "photosML")
            crop_size: Size of the cropped face image in pixels (default: 160)
            person_folder: If set, save to this folder with picture1.jpg, picture2.jpg, ... (for member data).
                          If None, save to output_folder with face_001.jpg, face_002.jpg, ...
        """
        # Create output folder if it doesn't exist
        if person_folder:
            output_path = Path(person_folder)
            use_picture_naming = True
        else:
            output_path = Path(output_folder)
            use_picture_naming = False
        output_path.mkdir(exist_ok=True)
        
        cap = self._open_camera()
        
        if cap is None:
            raise ValueError("Could not open webcam. Please check camera permissions.")
        
        print("=" * 60)
        print(f"Face capture started! Will capture {num_captures} images.")
        print(f"Saving to: {output_path.absolute()}")
        print("Press 'q' to quit")
        print("=" * 60)
        
        capture_count = 0
        last_capture_time = 0
        capture_interval = 1.0  # Minimum seconds between captures
        
        while capture_count < num_captures:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Create a copy for display (with bounding box) - original frame stays clean for saving
            display_frame = frame.copy()
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect face landmarks
            detection_result = self.face_landmarker.detect(mp_image)
            
            # Process the first detected face
            face_detected = False
            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                face_landmarks = detection_result.face_landmarks[0]
                face_detected = True
                
                # Get eye landmarks for alignment
                # Left eye center (landmark 33) and right eye center (landmark 263)
                left_eye = face_landmarks[33]  # Left eye outer corner
                right_eye = face_landmarks[263]  # Right eye outer corner
                
                # Convert to pixel coordinates
                left_eye_px = (int(left_eye.x * w), int(left_eye.y * h))
                right_eye_px = (int(right_eye.x * w), int(right_eye.y * h))
                
                # Calculate angle between eyes for rotation
                dy = right_eye_px[1] - left_eye_px[1]
                dx = right_eye_px[0] - left_eye_px[0]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Calculate center point between eyes
                eye_center = ((left_eye_px[0] + right_eye_px[0]) // 2,
                             (left_eye_px[1] + right_eye_px[1]) // 2)
                
                # Get face bounding box for visual display (preview only, not saved)
                xs = [landmark.x * w for landmark in face_landmarks]
                ys = [landmark.y * h for landmark in face_landmarks]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                
                # Draw bounding box on display frame only (green box for visual feedback)
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Auto-capture or manual capture
                current_time = time.time()
                
                # Auto-capture every interval if face is detected
                if current_time - last_capture_time >= capture_interval:
                    try:
                        # Calculate face size (use distance between eyes as reference)
                        eye_distance = np.sqrt(dx**2 + dy**2)
                        # Use a multiplier to get a good crop size (typically 2.5-3x eye distance)
                        # This ensures we get the full face including chin and forehead
                        face_crop_size = int(eye_distance * 3.0)
                        
                        # Ensure minimum size to avoid too small crops
                        face_crop_size = max(face_crop_size, crop_size * 2)
                        
                        # Use MediaPipe's facial transformation matrix for better alignment
                        # This handles 3D face orientation (pitch, yaw, roll) and provides more accurate alignment
                        if (detection_result.facial_transformation_matrixes and 
                            len(detection_result.facial_transformation_matrixes) > 0):
                            try:
                                # MediaPipe provides a 4x4 transformation matrix
                                transform_matrix = detection_result.facial_transformation_matrixes[0]
                                
                                # Convert to numpy array if it's not already
                                if not isinstance(transform_matrix, np.ndarray):
                                    transform_matrix = np.array(transform_matrix)
                                
                                # Handle both flat array (16 elements) and 2D array (4x4) formats
                                if transform_matrix.ndim == 1:
                                    # Flat array format: [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15]
                                    # Column-major order
                                    cos_theta = transform_matrix[0]   # R[0,0] = cos(roll)
                                    sin_theta = transform_matrix[4]   # R[0,1] = -sin(roll)
                                else:
                                    # 2D array format (4x4 matrix)
                                    # Access as [row, col]
                                    cos_theta = transform_matrix[0, 0]   # R[0,0] = cos(roll)
                                    sin_theta = transform_matrix[0, 1]   # R[0,1] = -sin(roll)
                                
                                # Extract roll angle (rotation around Z-axis)
                                roll_angle = np.degrees(np.arctan2(-sin_theta, cos_theta))
                                
                                # Use MediaPipe's roll angle for alignment (more accurate for 3D face poses)
                                alignment_angle = roll_angle
                            except (IndexError, AttributeError) as e:
                                # If matrix access fails, fall back to eye-based alignment
                                print(f"Warning: Could not extract roll angle from transformation matrix, using eye-based alignment: {e}")
                                alignment_angle = angle
                        else:
                            # Fallback to eye-based alignment (works well for 2D alignment)
                            alignment_angle = angle
                        
                        # Rotate image to align face (rotate around eye center)
                        rotation_matrix = cv2.getRotationMatrix2D(eye_center, alignment_angle, 1.0)
                        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h))
                        
                        # Transform eye center point after rotation
                        eye_center_homogeneous = np.array([eye_center[0], eye_center[1], 1])
                        eye_center_rotated = rotation_matrix @ eye_center_homogeneous
                        eye_center_rotated = (int(eye_center_rotated[0]), int(eye_center_rotated[1]))
                        
                        # Calculate crop region centered on rotated eye center
                        # Adjust crop center slightly downward to center on face (not just eyes)
                        crop_center_x = eye_center_rotated[0]
                        crop_center_y = eye_center_rotated[1] + int(face_crop_size * 0.1)  # Slight downward adjustment
                        
                        crop_x = max(0, crop_center_x - face_crop_size // 2)
                        crop_y = max(0, crop_center_y - face_crop_size // 2)
                        crop_x2 = min(w, crop_x + face_crop_size)
                        crop_y2 = min(h, crop_y + face_crop_size)
                        
                        # Ensure we have a square region
                        actual_width = crop_x2 - crop_x
                        actual_height = crop_y2 - crop_y
                        if actual_width != actual_height:
                            # Make it square by using the smaller dimension
                            size = min(actual_width, actual_height)
                            crop_x2 = crop_x + size
                            crop_y2 = crop_y + size
                        
                        # Crop the face region
                        cropped_face = rotated_frame[crop_y:crop_y2, crop_x:crop_x2]
                        
                        # Resize to target size
                        if cropped_face.size > 0 and cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                            resized_face = cv2.resize(cropped_face, (crop_size, crop_size))
                            
                            # Save the image (pictureN.jpg for person folders, face_NNN.jpg otherwise)
                            if use_picture_naming:
                                filename = output_path / f"picture{capture_count + 1}.jpg"
                            else:
                                filename = output_path / f"face_{capture_count + 1:03d}.jpg"
                            cv2.imwrite(str(filename), resized_face)
                            
                            capture_count += 1
                            last_capture_time = current_time
                            print(f"Captured image {capture_count}/{num_captures}: {filename}")
                            
                            # Visual feedback on display frame
                            cv2.putText(display_frame, f"CAPTURED! ({capture_count}/{num_captures})", 
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error capturing face: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Display status on display frame
            status_text = f"Captured: {capture_count}/{num_captures}"
            if face_detected:
                status_text += " - Face detected"
            else:
                status_text += " - No face detected"
            
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Capture - Aligned', display_frame)
            
            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopping capture...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nCapture complete! {capture_count} images saved to {output_path}/")

def main():
    parser = argparse.ArgumentParser(description='Extract face region and feature points using MediaPipe')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to save output image (optional)')
    parser.add_argument('--webcam', action='store_true', help='Use webcam instead of image')
    parser.add_argument('--capture', action='store_true', help='Capture aligned face images from webcam')
    parser.add_argument('--num-captures', type=int, default=20, help='Number of face images to capture (default: 20)')
    parser.add_argument('--output-folder', type=str, default='photosML', help='Folder to save captured images when --person is not set (default: photosML)')
    parser.add_argument('--person', type=str, default=None, help='Save to this person folder with picture1.jpg, picture2.jpg, ... (e.g. person7 for new member)')
    
    args = parser.parse_args()
    
    extractor = FaceExtractor()
    
    if args.capture:
        extractor.capture_aligned_faces(num_captures=args.num_captures, output_folder=args.output_folder, person_folder=args.person)
    elif args.webcam:
        extractor.extract_from_webcam()
    elif args.image:
        extractor.extract_from_image(args.image, args.output)
    else:
        # Try to use a sample image if available, or prompt user
        print("Usage:")
        print("  python face_extraction_mine_trial.py --image <image_path> [--output <output_path>]")
        print("  python face_extraction_mine_trial.py --webcam")
        print("  python face_extraction_mine_trial.py --capture [--num-captures N] [--output-folder FOLDER] [--person FOLDER]")
        print("\nExample (capture for a new member into person7/):")
        print("  python face_extraction_mine_trial.py --capture --person person7 --num-captures 20")
        print("\nExample (capture into default photosML/ with face_001.jpg, ...):")
        print("  python face_extraction_mine_trial.py --capture --num-captures 20 --output-folder photosML")

if __name__ == "__main__":
    main()
