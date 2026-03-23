"""
face_detector_haar.py - Real-Time Face Detection using Haar Cascade
"""

import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import initialize_camera, draw_face_box, draw_info_overlay, FPSCounter


def main():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("ERROR: Failed to load Haar Cascade classifier")
        return

    print("Haar Cascade classifier loaded")

    try:
        cap = initialize_camera(camera_index=0, width=640, height=480)
    except IOError as e:
        print(f"ERROR: {e}")
        return

    fps_counter = FPSCounter()

    print("Starting face detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for x, y, w, h in faces:
            draw_face_box(frame, x, y, w, h, label="Face")

        fps = fps_counter.update()
        draw_info_overlay(frame, len(faces), fps)

        cv2.imshow("Haar Cascade Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed gracefully.")


if __name__ == "__main__":
    main()
