# 🛠️ Step-by-Step Implementation Guide

> **Document:** 06 — Implementation  
> **Version:** 1.0  
> **Last Updated:** March 23, 2026  
> **Author:** Ankit (AlgorithmicMind)

---

## 📌 Overview

This document provides a complete, line-by-line walkthrough for building the Real-Time Face Detection System. We will build **two implementations** — one using Haar Cascade and one using DNN — plus a shared utilities module.

---

## 1. Project File Structure

Create these files inside the project root:

```
face_detect_bot/
├── src/
│   ├── face_detector_haar.py    # Haar Cascade version
│   ├── face_detector_dnn.py     # DNN version
│   └── utils.py                 # Helper utilities
├── models/                      # Pre-trained model files
├── requirements.txt             # Dependencies
└── docs/                        # Documentation (you are here)
```

### Create the directories:

```bash
mkdir src
mkdir models
```

---

## 2. Create `requirements.txt`

```txt
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80
numpy==1.26.4
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 3. Build `src/utils.py` — Helper Utilities

This module centralizes reusable functions.

```python
"""
utils.py — Shared utilities for the Face Detection System
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

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"✅ Camera initialized ({width}x{height})")
    return cap


def draw_face_box(frame, x, y, w, h, label=None,
                  color=(0, 255, 0), thickness=2):
    """
    Draw a rectangle around a detected face with an optional label.

    Args:
        frame: Image/frame to draw on
        x, y: Top-left corner coordinates
        w, h: Width and height of the bounding box
        label (str): Optional text label (e.g., confidence %)
        color (tuple): BGR color for the rectangle
        thickness (int): Line thickness
    """
    # Draw main rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Draw label background + text if provided
    if label:
        # Background for label
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )[0]
        cv2.rectangle(
            frame,
            (x, y - label_size[1] - 10),
            (x + label_size[0] + 5, y),
            color, -1  # Filled rectangle
        )
        # Text
        cv2.putText(
            frame, label, (x + 2, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 1, cv2.LINE_AA
        )


def draw_info_overlay(frame, face_count, fps=None):
    """
    Draw an information overlay on the frame showing face count and FPS.

    Args:
        frame: Image/frame to draw on
        face_count (int): Number of detected faces
        fps (float): Frames per second (optional)
    """
    h, w = frame.shape[:2]

    # Semi-transparent background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Face count text
    cv2.putText(
        frame,
        f"Faces Detected: {face_count}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2, cv2.LINE_AA
    )

    # FPS counter (right-aligned)
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )[0]
        cv2.putText(
            frame,
            fps_text,
            (w - text_size[0] - 10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2, cv2.LINE_AA
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
```

---

## 4. Build `src/face_detector_haar.py`

```python
"""
face_detector_haar.py — Real-Time Face Detection using Haar Cascade
"""

import cv2
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import initialize_camera, draw_face_box, draw_info_overlay, FPSCounter


def main():
    # ── Step 1: Load Haar Cascade Classifier ──
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("❌ ERROR: Failed to load Haar Cascade classifier")
        return

    print("✅ Haar Cascade classifier loaded")

    # ── Step 2: Initialize Webcam ──
    try:
        cap = initialize_camera(camera_index=0, width=640, height=480)
    except IOError as e:
        print(f"❌ {e}")
        return

    # ── Step 3: Initialize FPS Counter ──
    fps_counter = FPSCounter()

    print("🎥 Starting face detection... Press 'q' to quit.")
    print("-" * 50)

    # ── Step 4: Main Detection Loop ──
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam")
            break

        # Convert to grayscale (required by Haar Cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Improve contrast with histogram equalization
        gray = cv2.equalizeHist(gray)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw bounding boxes for each detected face
        for (x, y, w, h) in faces:
            draw_face_box(frame, x, y, w, h, label="Face")

        # Update FPS
        fps = fps_counter.update()

        # Draw info overlay
        draw_info_overlay(frame, len(faces), fps)

        # Display the frame
        cv2.imshow('Haar Cascade Face Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Step 5: Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Application closed gracefully.")


if __name__ == "__main__":
    main()
```

---

## 5. Build `src/face_detector_dnn.py`

```python
"""
face_detector_dnn.py — Real-Time Face Detection using DNN (SSD + ResNet-10)
"""

import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import initialize_camera, draw_face_box, draw_info_overlay, FPSCounter


def main():
    # ── Step 1: Load DNN Model ──
    prototxt = os.path.join("models", "deploy.prototxt")
    caffemodel = os.path.join(
        "models", "res10_300x300_ssd_iter_140000.caffemodel"
    )

    if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
        print("❌ Model files not found in 'models/' directory!")
        print("   Download them first. See docs/05_dnn_detection_guide.md")
        return

    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    print("✅ DNN model loaded successfully")

    # ── Step 2: Initialize Webcam ──
    try:
        cap = initialize_camera(camera_index=0, width=640, height=480)
    except IOError as e:
        print(f"❌ {e}")
        return

    # ── Config ──
    CONFIDENCE_THRESHOLD = 0.5
    fps_counter = FPSCounter()

    print("🎥 Starting DNN face detection... Press 'q' to quit.")
    print("-" * 50)

    # ── Step 3: Main Detection Loop ──
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]

        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False, crop=False
        )

        # Forward pass
        net.setInput(blob)
        detections = net.forward()

        # Process detections
        face_count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_THRESHOLD:
                face_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                bw, bh = x2 - x1, y2 - y1
                label = f"{confidence * 100:.1f}%"
                draw_face_box(frame, x1, y1, bw, bh, label=label)

        fps = fps_counter.update()
        draw_info_overlay(frame, face_count, fps)
        cv2.imshow('DNN Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Application closed gracefully.")


if __name__ == "__main__":
    main()
```

---

## 6. Running the Application

### Run Haar Cascade version:
```bash
cd face_detect_bot
python src/face_detector_haar.py
```

### Run DNN version:
```bash
cd face_detect_bot
python src/face_detector_dnn.py
```

### Controls:
| Key | Action |
|-----|--------|
| `q` | Quit the application |

---

## 7. Code Flow Summary

```
main()
  │
  ├── Load face detection model (Haar / DNN)
  ├── Initialize webcam via utils.initialize_camera()
  ├── Create FPSCounter instance
  │
  └── WHILE True:
        ├── Read frame from webcam
        ├── Preprocess frame (grayscale / blob)
        ├── Run detection (detectMultiScale / forward)
        ├── Loop through detections:
        │     └── Draw bounding box via utils.draw_face_box()
        ├── Update FPS counter
        ├── Draw overlay via utils.draw_info_overlay()
        ├── Display frame with cv2.imshow()
        └── Break if 'q' pressed
```

---

## ⏭️ Next Step

👉 Proceed to **[07 — Testing & Debugging](./07_testing_debugging.md)**
