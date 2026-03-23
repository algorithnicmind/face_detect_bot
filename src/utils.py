"""
utils.py - Shared utilities for the Face Detection System
"""

import cv2
import time


def initialize_camera(camera_index=0, width=640, height=480):
    """
    Initialize and configure the webcam.

    Args:
        camera_index (int): Camera device index (0 = default)
        width (int): Desired frame width
        height (int): Desired frame height

    Returns:
        cv2.VideoCapture: Configured camera object

    Raises:
        IOError: If camera cannot be opened
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise IOError(
            f"Cannot open camera at index {camera_index}. "
            "Check if webcam is connected and not used by another app."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"Camera initialized ({width}x{height})")
    return cap


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


def draw_info_overlay(frame, face_count, fps=None):
    """
    Draw an information overlay on the frame showing face count and FPS.
    """
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame,
        f"Faces Detected: {face_count}",
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
