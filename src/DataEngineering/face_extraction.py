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

    def _crop_square_and_resize(
        self,
        image: np.ndarray,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        *,
        padding_px: int = 20,
        output_size: tuple[int, int] = (160, 160),
        border_type: int = cv2.BORDER_REFLECT_101,
        border_value: int | tuple[int, int, int] = 0,
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
        """
        Crop a *square* region centered on the bbox and resize to output_size.

        This avoids distortion caused by resizing a non-square crop to 160x160.

        Returns:
            (face_resized_or_None, (sx1, sy1, sx2, sy2)) where coords are the
            unclipped square box in the original image coordinate system after
            applying padding (clipped coords are used internally for cropping).
        """
        if image is None or image.size == 0:
            return None, (0, 0, 0, 0)

        h, w = image.shape[:2]
        # Clamp input bbox to image bounds first.
        x_min = max(0, min(w, int(x_min)))
        x_max = max(0, min(w, int(x_max)))
        y_min = max(0, min(h, int(y_min)))
        y_max = max(0, min(h, int(y_max)))

        bw = max(0, x_max - x_min)
        bh = max(0, y_max - y_min)
        if bw == 0 or bh == 0:
            return None, (x_min, y_min, x_max, y_max)

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        side = max(bw, bh) + 2 * max(0, int(padding_px))
        side = max(1, int(round(side)))
        half = side / 2.0

        sx1 = int(round(cx - half))
        sy1 = int(round(cy - half))
        sx2 = sx1 + side
        sy2 = sy1 + side

        # Compute padding needed if the square goes out of bounds.
        pad_left = max(0, -sx1)
        pad_top = max(0, -sy1)
        pad_right = max(0, sx2 - w)
        pad_bottom = max(0, sy2 - h)

        cx1 = max(0, sx1)
        cy1 = max(0, sy1)
        cx2 = min(w, sx2)
        cy2 = min(h, sy2)

        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None, (sx1, sy1, sx2, sy2)

        if any(v > 0 for v in (pad_left, pad_top, pad_right, pad_bottom)):
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=border_type,
                value=border_value,
            )

        # Ensure crop is square (can be off by 1 due to rounding).
        ch, cw = crop.shape[:2]
        if ch != cw:
            side2 = max(ch, cw)
            extra_y = side2 - ch
            extra_x = side2 - cw
            crop = cv2.copyMakeBorder(
                crop,
                extra_y // 2,
                extra_y - (extra_y // 2),
                extra_x // 2,
                extra_x - (extra_x // 2),
                borderType=border_type,
                value=border_value,
            )

        out_w, out_h = int(output_size[0]), int(output_size[1])
        interp = cv2.INTER_AREA if (crop.shape[1] > out_w or crop.shape[0] > out_h) else cv2.INTER_LINEAR
        face_resized = cv2.resize(crop, (out_w, out_h), interpolation=interp)
        return face_resized, (sx1, sy1, sx2, sy2)
        
    def extract_from_image(self, image_path, output_path=None, output_folder=None):
        """
        Extract face region and feature points from an image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save output image (file path)
            output_folder: Optional folder to save output image (saves with original filename)
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
        if output_folder:
            # Save to folder with original filename
            folder_path = Path(output_folder)
            folder_path.mkdir(exist_ok=True)
            input_path = Path(image_path)
            output_file = folder_path / f"{input_path.stem}_annotated{input_path.suffix}"
            cv2.imwrite(str(output_file), annotated_image)
            print(f"\nOutput saved to: {output_file}")
        elif output_path:
            # Save to specified file path
            cv2.imwrite(output_path, annotated_image)
            print(f"\nOutput saved to: {output_path}")
        else:
            # Display the image
            cv2.imshow('Face Detection and Landmarks', annotated_image)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated_image, detection_result
    
    def extract_from_webcam(self, camera_index=0):
        """
        Extract face region and feature points from webcam feed.
        Press 'q' to quit.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        cap = cv2.VideoCapture(camera_index)
        
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
    
    def _get_capture_instruction(self, images_captured, num_images):
        """
        Get pose instruction based on capture progress.
        Returns (instruction_text, instruction_color).
        Colors are darker BGR for readability; each phase uses a distinct color.
        """
        # Calculate progress percentage
        progress = images_captured / num_images if num_images > 0 else 0
        
        # Phase 1: First 25% - Front facing, close up (dark teal/cyan)
        PHASE1_COLOR = (180, 140, 80)  # BGR: dark teal
        if progress < 0.25:
            phase_progress = images_captured / max(1, int(num_images * 0.25))
            if phase_progress < 0.5:
                return "Look straight at camera, close distance", PHASE1_COLOR
            else:
                return "Keep looking forward, stay close", PHASE1_COLOR
        
        # Phase 2: 25-50% - Slight angles, medium distance (dark green)
        PHASE2_COLOR = (80, 180, 100)  # BGR: dark green
        if progress < 0.50:
            phase_num = images_captured - int(num_images * 0.25)
            phase_total = int(num_images * 0.25)
            sub_progress = phase_num / max(1, phase_total)
            
            if sub_progress < 0.33:
                return "Tilt head slightly LEFT", PHASE2_COLOR
            elif sub_progress < 0.66:
                return "Tilt head slightly RIGHT", PHASE2_COLOR
            else:
                return "Move back a little, look forward", PHASE2_COLOR
        
        # Phase 3: 50-75% - More angles, varied distance (dark orange/rust)
        PHASE3_COLOR = (0, 100, 255)  # BGR: dark orange
        if progress < 0.75:
            phase_num = images_captured - int(num_images * 0.50)
            phase_total = int(num_images * 0.25)
            sub_progress = phase_num / max(1, phase_total)
            
            if sub_progress < 0.25:
                return "Turn head slightly LEFT, chin up", PHASE3_COLOR
            elif sub_progress < 0.50:
                return "Turn head slightly RIGHT, chin up", PHASE3_COLOR
            elif sub_progress < 0.75:
                return "Tilt head DOWN slightly", PHASE3_COLOR
            else:
                return "Move further back, face forward", PHASE3_COLOR
        
        # Phase 4: 75-100% - All angles, ensure features visible (dark magenta/purple)
        PHASE4_COLOR = (255, 60, 180)  # BGR: dark magenta
        phase_num = images_captured - int(num_images * 0.75)
        phase_total = num_images - int(num_images * 0.75)
        sub_progress = phase_num / max(1, phase_total)
        
        if sub_progress < 0.20:
            return "3/4 view LEFT (features visible!)", PHASE4_COLOR
        elif sub_progress < 0.40:
            return "3/4 view RIGHT (features visible!)", PHASE4_COLOR
        elif sub_progress < 0.60:
            return "Slight smile, look at camera", PHASE4_COLOR
        elif sub_progress < 0.80:
            return "Resting face, any comfortable angle", PHASE4_COLOR
        else:
            return "Almost done! Any natural pose", PHASE4_COLOR

    def capture_face_dataset(self, person_folder="person1", num_images=20, image_size=(160, 160), camera_index=0, manual_trigger=False):
        """
        Capture face images from webcam and save them to a folder.
        Takes specified number of images when face is detected.
        
        Args:
            person_folder: Folder name to save images (default: "person1")
            num_images: Number of images to capture (default: 20)
            image_size: Size of saved images as (width, height) (default: (160, 160))
            camera_index: Index of the camera to use (default: 0)
            manual_trigger: If True, capture only when spacebar is pressed (default: False)
        """
        import os
        import time
        
        # Create the folder if it doesn't exist
        folder_path = Path(person_folder)
        folder_path.mkdir(exist_ok=True)
        print(f"Images will be saved to: {folder_path.absolute()}")
        print(f"Using camera index: {camera_index}")
        print(f"Capture mode: {'Manual (press SPACE to capture)' if manual_trigger else 'Automatic (1 image/second)'}")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        images_captured = 0
        last_capture_time = 0
        capture_interval = 1.0  # Capture one image every 1 second (for auto mode)
        
        print("=" * 60)
        print(f"Face Dataset Capture Mode")
        print(f"Target: {num_images} images")
        print(f"Image size: {image_size[0]}x{image_size[1]} pixels")
        if manual_trigger:
            print(f"Press SPACE to capture, 'q' or 'ESC' to stop")
        else:
            print(f"Press 'q' or 'ESC' to stop")
        print("=" * 60)
        print("\nPosition yourself in front of the camera...")
        if not manual_trigger:
            print("Starting in 5 seconds...\n")
        else:
            print("Press SPACE when ready to capture.\n")
        
        # 5-second countdown before starting (only for auto mode)
        countdown_start = time.time()
        countdown_duration = 5.0 if not manual_trigger else 0.0
        
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

                # Crop a square region (prevents distortion when resizing to 160x160)
                face_resized, (sx1, sy1, sx2, sy2) = self._crop_square_and_resize(
                    frame,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    padding_px=20,
                    output_size=image_size,
                )

                # Draw the square crop area (clipped for display)
                dx1, dy1 = max(0, sx1), max(0, sy1)
                dx2, dy2 = min(w, sx2), min(h, sy2)
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
                
            # Display status on frame
            if remaining_countdown > 0:
                status_text = f"Starting in {int(remaining_countdown) + 1} seconds..."
                status_color = (0, 165, 255)  # Orange
            else:
                status_text = f"Captured: {images_captured}/{num_images}"
                if face_detected:
                    if manual_trigger:
                        status_text += " [Face Detected - Press SPACE]"
                    else:
                        status_text += " [Face Detected - Capturing...]"
                else:
                    status_text += " [No Face Detected]"
                status_color = (0, 255, 0)  # Green
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Display pose instruction (only after countdown and when capturing)
            if can_capture:
                instruction, instr_color = self._get_capture_instruction(images_captured, num_images)
                if instruction:
                    # Draw background rectangle behind instruction for clarity
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (tw, th), baseline = cv2.getTextSize(instruction, font, font_scale, thickness)
                    padding = 8
                    instr_y = 60  # Baseline position for instruction text
                    x1, y1 = 10, instr_y - th - padding
                    x2, y2 = 10 + tw + padding * 2, instr_y + baseline + padding
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)  # Dark gray background
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)  # Light border
                    cv2.putText(frame, instruction, (10 + padding, instr_y), 
                               font, font_scale, instr_color, thickness)
            
            if manual_trigger:
                cv2.putText(frame, "SPACE: capture | 'q'/ESC: stop", (10, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Press 'q' or 'ESC' to stop", (10, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Dataset Capture', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Handle spacebar for manual capture
            space_pressed = (key == ord(' '))
            
            # Capture logic
            if can_capture and face_detected and face_resized is not None and face_resized.size > 0:
                should_capture = False
                if manual_trigger:
                    # Manual mode: capture only when space is pressed
                    if space_pressed:
                        should_capture = True
                else:
                    # Auto mode: capture if enough time has passed
                    if current_time - last_capture_time >= capture_interval:
                        should_capture = True
                
                if should_capture:
                    # Save the image
                    image_filename = folder_path / f"picture{images_captured + 1}.jpg"
                    cv2.imwrite(str(image_filename), face_resized)
                    
                    images_captured += 1
                    last_capture_time = current_time
                    
                    print(f"Captured image {images_captured}/{num_images}: {image_filename.name}")
            
            # Check for quit keys
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
    
    def process_existing_images(self, input_path, output_folder="processed_faces", 
                                image_size=(160, 160), resize=True):
        """
        Process existing images and extract face regions with bounding boxes.
        Can process a single image or all images in a folder.
        
        Args:
            input_path: Path to a single image file or folder containing images
            output_folder: Folder name to save processed face images (default: "processed_faces")
            image_size: Target size for saved images as (width, height) (default: (160, 160))
            resize: Whether to resize extracted faces to image_size (default: True)
        """
        import os
        
        input_path = Path(input_path)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Get list of images to process
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images_to_process = []
        
        if input_path.is_file():
            # Single image
            if input_path.suffix.lower() in image_extensions:
                images_to_process = [input_path]
            else:
                raise ValueError(f"File {input_path} is not a supported image format")
        elif input_path.is_dir():
            # Folder - get all images
            for ext in image_extensions:
                images_to_process.extend(input_path.glob(f'*{ext}'))
                images_to_process.extend(input_path.glob(f'*{ext.upper()}'))
            images_to_process.sort()
        else:
            raise ValueError(f"Path {input_path} does not exist")
        
        if not images_to_process:
            print(f"No images found in {input_path}")
            return
        
        print("=" * 60)
        print(f"Processing {len(images_to_process)} image(s)")
        print(f"Output folder: {output_path.absolute()}")
        print(f"Resize: {resize}, Size: {image_size[0]}x{image_size[1] if resize else 'original'}")
        print("=" * 60)
        
        total_faces = 0
        processed_images = 0
        
        for img_path in images_to_process:
            print(f"\nProcessing: {img_path.name}")
            
            # Read image - use numpy to handle unicode paths
            try:
                # Try reading with cv2 first
                image = cv2.imread(str(img_path))
                # If that fails, try reading with numpy for unicode support
                if image is None:
                    import numpy as np
                    img_array = np.fromfile(str(img_path), np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"  [WARNING] Error reading image: {img_path.name} - {str(e)}")
                continue
            
            if image is None:
                print(f"  [WARNING] Could not read image: {img_path.name}")
                continue
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Detect face landmarks
            detection_result = self.face_landmarker.detect(mp_image)
            
            if not detection_result.face_landmarks:
                print(f"  [WARNING] No faces detected in: {img_path.name}")
                continue
            
            # Process each detected face
            for face_idx, face_landmarks in enumerate(detection_result.face_landmarks):
                # Get bounding box from landmarks
                xs = [landmark.x * w for landmark in face_landmarks]
                ys = [landmark.y * h for landmark in face_landmarks]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))

                if resize:
                    face_processed, _ = self._crop_square_and_resize(
                        image,
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        padding_px=20,
                        output_size=image_size,
                    )
                else:
                    # Crop a square region but DON'T resize (keeps pixel geometry 1:1).
                    bw = max(1, x_max - x_min)
                    bh = max(1, y_max - y_min)
                    side = max(bw, bh) + 2 * 20
                    face_processed, _ = self._crop_square_and_resize(
                        image,
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        padding_px=20,
                        output_size=(side, side),
                    )

                if face_processed is None or face_processed.size == 0:
                    continue
                
                # Generate output filename
                base_name = img_path.stem
                if len(detection_result.face_landmarks) > 1:
                    # Multiple faces - add face number
                    output_filename = output_path / f"{base_name}_face{face_idx + 1}.jpg"
                else:
                    # Single face - use original name
                    output_filename = output_path / f"{base_name}.jpg"
                
                # Save the processed face
                cv2.imwrite(str(output_filename), face_processed)
                total_faces += 1
                print(f"  [OK] Saved face {face_idx + 1}: {output_filename.name}")
            
            processed_images += 1
        
        print("\n" + "=" * 60)
        print(f"Processing complete!")
        print(f"  Images processed: {processed_images}/{len(images_to_process)}")
        print(f"  Total faces extracted: {total_faces}")
        print(f"  Output folder: {output_path.absolute()}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Extract face region and feature points using MediaPipe')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to save output image file (optional)')
    parser.add_argument('--output-folder', type=str, help='Folder to save output image (saves with original filename + _annotated)')
    parser.add_argument('--webcam', action='store_true', help='Use webcam instead of image')
    parser.add_argument('--capture', action='store_true', help='Capture face dataset (20 images, 160x160)')
    parser.add_argument('--person', type=str, default='person1', help='Output folder name for dataset capture (default: person1)')
    parser.add_argument('--num-images', type=int, default=20, help='Number of images to capture (default: 20)')
    parser.add_argument('--process', type=str, help='Process existing images: path to image file or folder')
    parser.add_argument('--process-output', type=str, default='processed_faces', help='Output folder for processed images (default: processed_faces)')
    parser.add_argument('--no-resize', action='store_true', help='Keep original size when processing images (default: resize to 160x160)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0, use 1 for second webcam)')
    parser.add_argument('--manual', action='store_true', help='Manual capture mode: press SPACE to take each picture (default: auto capture)')
    
    args = parser.parse_args()
    
    extractor = FaceExtractor()
    
    if args.process:
        resize = not args.no_resize
        extractor.process_existing_images(
            input_path=args.process,
            output_folder=args.process_output,
            resize=resize
        )
    elif args.capture:
        extractor.capture_face_dataset(person_folder=args.person, num_images=args.num_images, camera_index=args.camera, manual_trigger=args.manual)
    elif args.webcam:
        extractor.extract_from_webcam(camera_index=args.camera)
    elif args.image:
        extractor.extract_from_image(args.image, args.output, args.output_folder)
    else:
        # Try to use a sample image if available, or prompt user
        print("Usage:")
        print("  python face_extraction.py --image <image_path> [--output <file_path>] [--output-folder <folder>]")
        print("  python face_extraction.py --webcam")
        print("  python face_extraction.py --capture [--person <folder_name>] [--num-images <number>] [--camera <index>] [--manual]")
        print("  python face_extraction.py --process <image_or_folder> [--process-output <folder>] [--no-resize]")
        print("\nExample:")
        print("  python face_extraction.py --image photo.jpg --output result.jpg")
        print("  python face_extraction.py --image photo.jpg --output-folder results/")
        print("  python face_extraction.py --capture --person person1 --num-images 20 --camera 1 --manual")
        print("  python face_extraction.py --process person1/ --process-output extracted_faces")
        print("  python face_extraction.py --process photo1.jpg --process-output faces --no-resize")

if __name__ == "__main__":
    main()
