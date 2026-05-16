# File: face_detection.py
# Authors: ML Engineering Team
# Description: Face and eye detection using Haar Cascade Classifiers
#              This is the main algorithm for the Facial Recognition Attendance Checker

# IMPORTS
import cv2 as cv
import argparse
import sys

# GLOBAL VARIABLES
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# Cascade classifiers (initialized in main)
face_cascade = None
eyes_cascade = None


def detectAndDisplay(frame):
    """
    Detect faces and eyes in a frame and draw detection boxes.

    Parameters:
        frame: Input frame from video capture (BGR format)

    Returns:
        None (modifies frame in place and displays it)
    """
    # Convert to grayscale for detection
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Equalize histogram to improve detection
    frame_gray = cv.equalizeHist(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Calculate center point
        center = (x + w//2, y + h//2)
        # Draw ellipse around face
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        # Extract region of interest (ROI) for face
        faceROI = frame_gray[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            # Calculate eye center relative to original frame
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            # Draw circle around eye
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)

    # Display the resulting frame
    cv.imshow('Capture - Face detection', frame)


def main():
    """
    Main entry point for face detection program.
    Initializes cascades, opens camera, and processes video stream.

    Returns:
        EXIT_SUCCESS (0) on success, EXIT_FAILURE (1) on failure
    """
    global face_cascade, eyes_cascade

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade',
                        help='Path to face cascade.',
                        default=cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    parser.add_argument('--eyes_cascade',
                        help='Path to eyes cascade.',
                        default=cv.data.haarcascades + 'haarcascade_eye.xml')
    parser.add_argument('--camera',
                        help='Camera device number.',
                        type=int,
                        default=0)
    args = parser.parse_args()

    # Get cascade file paths
    face_cascade_name = args.face_cascade
    eyes_cascade_name = args.eyes_cascade

    # Initialize cascade classifiers
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()

    # Load the face cascade
    if not face_cascade.load(face_cascade_name):
        print(f'--(!)Error loading face cascade from: {face_cascade_name}')
        return EXIT_FAILURE

    # Load the eyes cascade
    if not eyes_cascade.load(eyes_cascade_name):
        print(f'--(!)Error loading eyes cascade from: {eyes_cascade_name}')
        return EXIT_FAILURE

    print('--Cascades loaded successfully')

    # Get camera device number
    camera_device = args.camera

    # Open video capture
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened():
        print('--(!)Error opening video capture')
        return EXIT_FAILURE

    print(f'--Camera {camera_device} opened successfully')
    print('--Press ESC to exit')

    # Main video processing loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame was captured successfully
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        # Detect and display faces/eyes
        detectAndDisplay(frame)

        # Wait for ESC key (key code 27) to exit
        if cv.waitKey(10) == 27:
            print('--ESC pressed, exiting...')
            break

    # Release resources
    cap.release()
    cv.destroyAllWindows()

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
