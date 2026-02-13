"""
Test script for face recognition pipeline
Use this to test the system with single images before running live recognition
"""

import cv2
from face_recognition_pipeline import FaceRecognitionPipeline
from pathlib import Path
import sys

def test_single_image(pipeline, image_path):
    """Test recognition on a single image"""
    print(f"\nTesting image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Get faces
    faces = pipeline.app.get(image)
    
    if len(faces) == 0:
        print("No faces detected in image")
        return
    
    print(f"Detected {len(faces)} face(s)")
    
    # Process each face
    for idx, face in enumerate(faces):
        person_name, confidence = pipeline.recognize_face(face.embedding)
        print(f"Face {idx + 1}: {person_name} (confidence: {confidence:.3f})")
        
        # Draw on image
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{person_name}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display result
    cv2.imshow('Test Result', image)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_database_info(pipeline):
    """Display information about the current database"""
    print("\n" + "="*50)
    print("DATABASE INFORMATION")
    print("="*50)
    
    if len(pipeline.face_database) == 0:
        print("Database is empty!")
        print("Please build the database first using the main script.")
        return False
    
    print(f"Number of people in database: {len(pipeline.face_database)}")
    print(f"Recognition threshold: {pipeline.threshold}")
    print("\nPeople in database:")
    for idx, name in enumerate(pipeline.face_database.keys(), 1):
        print(f"  {idx}. {name}")
    
    print("="*50)
    return True

def test_embedding_extraction(pipeline, image_path):
    """Test if embeddings can be extracted from an image"""
    print(f"\nTesting embedding extraction from: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image")
        return False
    
    embedding = pipeline.extract_embedding(image)
    
    if embedding is None:
        print("❌ Failed to extract embedding (no face detected)")
        return False
    else:
        print(f"✓ Successfully extracted embedding (dimension: {len(embedding)})")
        return True

def main():
    print("="*50)
    print("FACE RECOGNITION PIPELINE - TEST SCRIPT")
    print("="*50)
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = FaceRecognitionPipeline()
    
    # Test 1: Check database
    print("\n[TEST 1] Checking database...")
    has_database = test_database_info(pipeline)
    
    if not has_database:
        print("\nPlease run 'python3 face_recognition_pipeline.py' first to build the database.")
        return
    
    # Test 2: Test with sample images
    print("\n[TEST 2] Testing with images...")
    print("\nOptions:")
    print("1. Test with a single image")
    print("2. Test embedding extraction")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        image_path = input("Enter path to test image: ")
        if Path(image_path).exists():
            test_single_image(pipeline, image_path)
        else:
            print(f"Error: File {image_path} not found")
    
    elif choice == '2':
        image_path = input("Enter path to test image: ")
        if Path(image_path).exists():
            test_embedding_extraction(pipeline, image_path)
        else:
            print(f"Error: File {image_path} not found")
    
    elif choice == '3':
        print("Exiting...")
        return
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
