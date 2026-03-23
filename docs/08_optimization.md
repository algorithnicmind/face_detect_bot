# ⚡ Performance Optimization Guide

> **Document:** 08 — Optimization  
> **Version:** 1.0  
> **Last Updated:** March 23, 2026  
> **Author:** Ankit (AlgorithmicMind)

---

## 📌 Overview

This document covers techniques to improve the speed, accuracy, and resource usage of the face detection system for smooth real-time performance.

---

## 1. Frame Resizing (Biggest Impact)

Processing smaller frames dramatically speeds up detection:

```python
# Resize frame before detection (keep original for display)
scale_percent = 50  # Reduce to 50%
small_frame = cv2.resize(frame, None,
                          fx=scale_percent/100,
                          fy=scale_percent/100)

# Detect on small frame
gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# Scale coordinates back to original size
scale = 100 / scale_percent
for (x, y, w, h) in faces:
    x, y, w, h = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

**Impact:**

| Resolution | Detection Time | FPS Gain |
|------------|---------------|----------|
| 640×480 (full) | ~33ms | Baseline |
| 320×240 (50%) | ~10ms | ~3x faster |
| 160×120 (25%) | ~4ms | ~8x faster |

---

## 2. Skip Frame Detection

Don't run detection on every single frame — detect every N frames and reuse results:

```python
frame_count = 0
DETECT_EVERY_N = 3  # Detect every 3rd frame
last_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % DETECT_EVERY_N == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw using cached results
    for (x, y, w, h) in last_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 3. ROI (Region of Interest) Detection

If faces are expected in a certain area, limit detection to that region:

```python
h, w = frame.shape[:2]

# Only scan the center 60% of the frame
margin_x = int(w * 0.2)
margin_y = int(h * 0.2)
roi = frame[margin_y:h-margin_y, margin_x:w-margin_x]

gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_roi, 1.1, 5)

# Adjust coordinates back to full frame
for (x, y, fw, fh) in faces:
    cv2.rectangle(frame,
                  (x + margin_x, y + margin_y),
                  (x + margin_x + fw, y + margin_y + fh),
                  (0, 255, 0), 2)
```

---

## 4. Histogram Equalization

Improves detection in poor lighting by normalizing brightness:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Standard equalization
equalized = cv2.equalizeHist(gray)

# CLAHE (better for uneven lighting)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

faces = face_cascade.detectMultiScale(enhanced, 1.1, 5)
```

**Comparison:**

| Method | Best For |
|--------|----------|
| No equalization | Well-lit environments |
| `equalizeHist` | Uniformly dark/bright |
| **CLAHE** | **Uneven lighting (recommended)** |

---

## 5. Threaded Video Capture

Separate capture and processing into threads to avoid blocking:

```python
import cv2
from threading import Thread


class VideoCaptureThread:
    """Threaded video capture for non-blocking frame reading."""

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.running = True

        # Start background thread
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# Usage:
cap = VideoCaptureThread(0)
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    # ... detection code ...
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
```

---

## 6. DNN Backend Optimization

### OpenCV DNN Targets:

```python
# CPU (default)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# OpenCL (GPU acceleration on Intel/AMD)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# CUDA (NVIDIA GPU — requires OpenCV with CUDA build)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

---

## 7. Optimize Haar Parameters

| Scenario | scaleFactor | minNeighbors | minSize |
|----------|-------------|--------------|---------|
| Max speed | 1.3 | 3 | (50,50) |
| **Balanced** | **1.1** | **5** | **(30,30)** |
| Max accuracy | 1.05 | 7 | (20,20) |

---

## 8. Bounding Box Smoothing

Reduce jittery boxes by averaging across frames:

```python
from collections import deque

face_history = deque(maxlen=5)  # Keep last 5 detections

def smooth_boxes(current_faces):
    """Average bounding boxes across recent frames."""
    face_history.append(current_faces)

    if len(face_history) < 2:
        return current_faces

    # Simple: return the most recent detection
    # Advanced: implement IoU-based tracking and averaging
    return current_faces
```

---

## 9. Optimization Summary

| Technique | FPS Gain | Complexity | Recommended |
|-----------|----------|------------|-------------|
| Frame resize (50%) | ~3x | Easy | ✅ Yes |
| Skip frames (every 3rd) | ~2x | Easy | ✅ Yes |
| ROI detection | ~2x | Medium | Situational |
| CLAHE equalization | Accuracy↑ | Easy | ✅ Yes |
| Threaded capture | ~1.5x | Medium | ✅ Yes |
| OpenCL backend | ~2x | Easy | If available |
| Parameter tuning | Variable | Easy | ✅ Yes |
| Box smoothing | Visual↑ | Medium | ✅ Yes |

---

## ⏭️ Next Step

👉 Proceed to **[09 — Deployment Guide](./09_deployment_guide.md)**
