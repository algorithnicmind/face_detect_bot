"""
face_detector_haar.py - Real-Time Face Detection using Haar Cascade
"""

import cv2
import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    HAAR_SCALE_FACTOR,
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
    HAAR_MAX_SIZE,
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
    ROISelector,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Haar Cascade Face Detection")
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
        "--scale-factor",
        type=float,
        default=HAAR_SCALE_FACTOR,
        help=f"Scale factor for detection (default: {HAAR_SCALE_FACTOR})",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=HAAR_MIN_NEIGHBORS,
        help=f"Min neighbors (default: {HAAR_MIN_NEIGHBORS})",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        nargs=2,
        default=list(HAAR_MIN_SIZE),
        help=f"Min face size (default: {HAAR_MIN_SIZE})",
    )
    parser.add_argument(
        "--no-threaded", action="store_true", help="Disable threaded capture"
    )
    parser.add_argument("--roi", action="store_true", help="Select region of interest")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        logger.error("Failed to load Haar Cascade classifier")
        return

    logger.info("Haar Cascade classifier loaded")

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
    frame_count = 0
    skip_frames = 0  # Set to > 0 to skip frames
    roi = None

    # ROI selection
    if args.roi:
        ret, sample_frame = read_frame(cap)
        if ret and sample_frame is not None:
            selector = ROISelector("Select Detection Area")
            roi = selector.select(sample_frame)

    logger.info("Starting face detection... Press 'q' to quit.")

    while True:
        ret, frame = read_frame(cap)
        if not ret or frame is None:
            continue

        frame_count += 1
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            cv2.imshow("Haar Cascade Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Apply ROI if selected
        detect_frame, offset = ROISelector.apply_roi(frame, roi)
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=tuple(args.min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Draw ROI rectangle
        if roi:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        for x, y, w, h in faces:
            # Offset coordinates back to full frame
            draw_face_box(frame, x + offset[0], y + offset[1], w, h, label="Face")

        fps = fps_counter.update()
        draw_info_overlay(frame, len(faces), fps)

        cv2.imshow("Haar Cascade Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    release_camera(cap)
    cv2.destroyAllWindows()
    logger.info("Application closed.")


if __name__ == "__main__":
    main()
