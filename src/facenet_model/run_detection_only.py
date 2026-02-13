"""
Face detection only — no database, no recognition.
Uses the same InsightFace detector; draws bounding boxes only.
Press 'q' to quit.
"""

import cv2
import argparse
from insightface.app import FaceAnalysis


def main():
    parser = argparse.ArgumentParser(description="Run face detection only (no recognition)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--det-size", type=int, default=320, help="Detection size (default: 320, use 240 for faster)")
    parser.add_argument("--no-display", action="store_true", help="Run without opening a window (e.g. for benchmarking)")
    args = parser.parse_args()

    print("Loading face detection model...")
    app = FaceAnalysis(
        name="buffalo_sc",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
    print("Detection only — no database or recognition. Press 'q' to quit.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not args.no_display:
            cv2.imshow("Face detection only", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
