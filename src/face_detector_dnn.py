"""
face_detector_dnn.py - Real-Time Face Detection using DNN (SSD + ResNet-10)
"""

import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import initialize_camera, draw_face_box, draw_info_overlay, FPSCounter


def main():
    prototxt = os.path.join("models", "deploy.prototxt")
    caffemodel = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
        print("Model files not found in 'models/' directory!")
        print("Download them first. See docs/05_dnn_detection_guide.md")
        return

    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    print("DNN model loaded successfully")

    try:
        cap = initialize_camera(camera_index=0, width=640, height=480)
    except IOError as e:
        print(f"ERROR: {e}")
        return

    CONFIDENCE_THRESHOLD = 0.5
    fps_counter = FPSCounter()

    print("Starting DNN face detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
        )

        net.setInput(blob)
        detections = net.forward()

        face_count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_THRESHOLD:
                face_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                bw, bh = x2 - x1, y2 - y1
                label = f"{confidence * 100:.1f}%"
                draw_face_box(frame, x1, y1, bw, bh, label=label)

        fps = fps_counter.update()
        draw_info_overlay(frame, face_count, fps)
        cv2.imshow("DNN Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed gracefully.")


if __name__ == "__main__":
    main()
