"""
face_detector_dnn.py - Real-Time Face Detection using DNN (SSD + ResNet-10)
"""

import cv2
import numpy as np
import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    DNN_CONFIDENCE_THRESHOLD,
    DNN_INPUT_SIZE,
    DNN_MEAN_SUBTRACTION,
    MODELS_DIR,
    DNN_PROTOTXT,
    DNN_CAFFEMODEL,
    LOG_LEVEL,
    LOG_FORMAT,
)
from src.utils import (
    initialize_camera,
    read_frame,
    release_camera,
    draw_face_box,
    draw_info_overlay,
    FPSCounter,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DNN Face Detection (SSD + ResNet)")
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"Camera index (default: {DEFAULT_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help=f"Frame width (default: {DEFAULT_FRAME_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help=f"Frame height (default: {DEFAULT_FRAME_HEIGHT})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DNN_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold 0.0-1.0 (default: {DNN_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--no-threaded", action="store_true", help="Disable threaded capture"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load DNN Model
    prototxt = os.path.join(MODELS_DIR, DNN_PROTOTXT)
    caffemodel = os.path.join(MODELS_DIR, DNN_CAFFEMODEL)

    if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
        logger.error(f"Model files not found in '{MODELS_DIR}/' directory!")
        return

    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    logger.info("DNN model loaded successfully")

    # Initialize camera
    try:
        cap = initialize_camera(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            threaded=not args.no_threaded,
        )
    except IOError as e:
        logger.error(f"Camera error: {e}")
        return

    fps_counter = FPSCounter()
    logger.info("Starting DNN face detection... Press 'q' to quit.")

    while True:
        ret, frame = read_frame(cap)
        if not ret or frame is None:
            continue

        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1.0, DNN_INPUT_SIZE, DNN_MEAN_SUBTRACTION, swapRB=False, crop=False
        )

        net.setInput(blob)
        detections = net.forward()

        face_count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args.threshold:
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

    release_camera(cap)
    cv2.destroyAllWindows()
    logger.info("Application closed.")


if __name__ == "__main__":
    main()
