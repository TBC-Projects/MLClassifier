"""
Webcam loop: detect face, recognize member, mark attendance with debouncing.
Displays "Member: <name>", "Not in club", or "No face". Does not modify face_extraction.py.
"""
import argparse
from pathlib import Path

import cv2

from attendance import get_attendance_logger
from member_db import load_members
from recognize import FaceRecognizer


def main():
    parser = argparse.ArgumentParser(description="Webcam attendance: recognize members and log attendance")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory with knn_face_model.joblib and label_to_member_id.json")
    parser.add_argument("--face-model", type=str, default="face_landmarker.task", help="Path to face_landmarker.task")
    parser.add_argument("--attendance", type=str, default="attendance.json", help="Path to attendance log file")
    parser.add_argument("--debounce", type=float, default=30.0, help="Seconds before same member can be logged again")
    parser.add_argument("--distance-threshold", type=float, default=30.0, help="Max distance to accept as member (higher = more permissive)")
    parser.add_argument("--ratio-threshold", type=float, default=1.0, help="Reject if nearest/2nd-nearest distance ratio > this (1.0 = disabled)")
    args = parser.parse_args()

    recognizer = FaceRecognizer(
        model_dir=args.model_dir,
        model_path=args.face_model,
        distance_threshold=args.distance_threshold,
        ratio_threshold=args.ratio_threshold,
    )
    logger = get_attendance_logger(attendance_path=args.attendance, debounce_seconds=args.debounce)
    members = {m["member_id"]: m["name"] for m in load_members()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return 1

    print("Webcam attendance started. Press 'q' or ESC to quit.")
    print(f"Attendance log: {args.attendance}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        member_id, distance = recognizer.recognize(frame)
        if member_id is not None:
            name = members.get(member_id, member_id)
            label = f"Member: {name}"
            if logger.mark_present(member_id):
                label += " [Attendance logged]"
            color = (0, 255, 0)
        elif distance is not None:
            label = "Not in club"
            color = (0, 0, 255)
        else:
            label = "No face"
            color = (0, 0, 255)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "Press 'q' or ESC to quit", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Attendance", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    exit(main())
