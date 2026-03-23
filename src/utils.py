"""
utils.py - Shared utilities for the Face Detection System
"""

import cv2
import time
import logging
import threading
from collections import deque

from src.config import (
    LOG_LEVEL,
    LOG_FORMAT,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    INFO_OVERLAY_HEIGHT,
    INFO_OVERLAY_OPACITY,
    USE_THREADED_CAPTURE,
    CAPTURE_QUEUE_SIZE,
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ThreadedCamera:
    """Threaded video capture for non-blocking frame reading."""

    def __init__(self, camera_index=0, width=640, height=480, queue_size=2):
        self.cap = cv2.VideoCapture(camera_index)
        self.stopped = False
        self.queue = deque(maxlen=queue_size)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera at index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        logger.info(f"Threaded camera initialized ({width}x{height})")

    def start(self):
        """Start the background thread."""
        thread = threading.Thread(target=self._update, daemon=True)
        thread.start()
        return self

    def _update(self):
        """Continuously read frames in background."""
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.queue.append(frame)

    def read(self):
        """Get the latest frame."""
        if len(self.queue) > 0:
            return self.queue[-1]
        return None

    def stop(self):
        """Stop the thread and release camera."""
        self.stopped = True
        self.cap.release()
        logger.info("Threaded camera stopped")


def initialize_camera(
    camera_index=DEFAULT_CAMERA_INDEX,
    width=DEFAULT_FRAME_WIDTH,
    height=DEFAULT_FRAME_HEIGHT,
    threaded=USE_THREADED_CAPTURE,
):
    """
    Initialize and configure the webcam.

    Args:
        camera_index (int): Camera device index (0 = default)
        width (int): Desired frame width
        height (int): Desired frame height
        threaded (bool): Use threaded capture for better performance

    Returns:
        VideoCapture or ThreadedCamera object

    Raises:
        IOError: If camera cannot be opened
    """
    if threaded:
        camera = ThreadedCamera(camera_index, width, height)
        camera.start()
        return camera
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(
                f"Cannot open camera at index {camera_index}. "
                "Check if webcam is connected and not used by another app."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.info(f"Camera initialized ({width}x{height})")
        return cap


def read_frame(cap):
    """Read a frame from camera (standard or threaded)."""
    if isinstance(cap, ThreadedCamera):
        frame = cap.read()
        return (frame is not None, frame)
    return cap.read()


def release_camera(cap):
    """Release camera resources."""
    if isinstance(cap, ThreadedCamera):
        cap.stop()
    else:
        cap.release()
    logger.info("Camera released")


def draw_face_box(frame, x, y, w, h, label=None, color=(0, 255, 0), thickness=2):
    """
    Draw a rectangle around a detected face with an optional label.
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    if label:
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(
            frame, (x, y - label_size[1] - 10), (x + label_size[0] + 5, y), color, -1
        )
        cv2.putText(
            frame,
            label,
            (x + 2, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def draw_info_overlay(
    frame,
    face_count,
    fps=None,
    overlay_height=INFO_OVERLAY_HEIGHT,
    opacity=INFO_OVERLAY_OPACITY,
):
    """
    Draw an information overlay on the frame showing face count and FPS.
    """
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(
            frame,
            fps_text,
            (w - text_size[0] - 10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


class FPSCounter:
    """Simple FPS counter using time-based averaging."""

    def __init__(self, avg_frames=30):
        self.avg_frames = avg_frames
        self.timestamps = []
        self.fps = 0.0

    def update(self):
        """Call once per frame to update FPS."""
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.avg_frames:
            self.timestamps.pop(0)
        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            self.fps = (len(self.timestamps) - 1) / elapsed
        return self.fps
