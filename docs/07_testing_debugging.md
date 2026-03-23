# 🧪 Testing & Debugging Guide

> **Document:** 07 — Testing & Debugging  
> **Version:** 1.0  
> **Last Updated:** March 23, 2026  
> **Author:** Ankit (AlgorithmicMind)

---

## 📌 Overview

This document covers how to test each component of the face detection system, common bugs you may encounter, and how to debug them effectively.

---

## 1. Component-Level Testing

### 1.1 Test Webcam Access

```python
# test_webcam.py
import cv2

def test_webcam():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Cannot open webcam"

    ret, frame = cap.read()
    assert ret, "Cannot read frame"
    assert frame is not None, "Frame is None"
    assert frame.shape[2] == 3, "Frame should have 3 channels (BGR)"

    print(f"✅ Webcam OK — Resolution: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()

if __name__ == "__main__":
    test_webcam()
```

---

### 1.2 Test Haar Cascade Loading

```python
# test_haar.py
import cv2

def test_haar_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    assert not face_cascade.empty(), "Haar Cascade failed to load"
    print("✅ Haar Cascade loaded successfully")

if __name__ == "__main__":
    test_haar_cascade()
```

---

### 1.3 Test DNN Model Loading

```python
# test_dnn.py
import cv2
import os

def test_dnn_model():
    proto = "models/deploy.prototxt"
    model = "models/res10_300x300_ssd_iter_140000.caffemodel"

    assert os.path.exists(proto), f"Missing: {proto}"
    assert os.path.exists(model), f"Missing: {model}"

    net = cv2.dnn.readNetFromCaffe(proto, model)
    assert net is not None, "DNN model failed to load"
    print("✅ DNN model loaded successfully")

if __name__ == "__main__":
    test_dnn_model()
```

---

### 1.4 Test Detection on a Static Image

```python
# test_detection.py
import cv2

def test_detection_on_image(image_path):
    """Test face detection on a static image instead of live video."""
    img = cv2.imread(image_path)
    assert img is not None, f"Cannot read image: {image_path}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    print(f"✅ Detected {len(faces)} face(s) in {image_path}")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Test Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_detection_on_image("test_face.jpg")
```

---

## 2. Common Errors & Solutions

### ❌ Error: `ModuleNotFoundError: No module named 'cv2'`

**Cause:** OpenCV is not installed or virtual environment is not activated.

**Fix:**
```bash
# Activate virtual environment first
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux

# Then install
pip install opencv-python
```

---

### ❌ Error: `Cannot open camera` / Black screen

**Cause:** Webcam is in use by another app, or wrong camera index.

**Fix:**
```python
# Try different camera indices
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        cap.release()
```

---

### ❌ Error: `Haar Cascade returns empty results`

**Cause:** Parameters are too strict, or input is not grayscale.

**Fix checklist:**
- [x] Convert to grayscale before detection
- [x] Use `scaleFactor=1.1` and `minNeighbors=3`
- [x] Ensure face is large enough (> `minSize`)
- [x] Check lighting conditions

---

### ❌ Error: `DNN blob returns no detections`

**Cause:** Model files are corrupted or wrong mean values.

**Fix:**
```python
# Verify model file sizes
import os
proto_size = os.path.getsize("models/deploy.prototxt")
model_size = os.path.getsize("models/res10_300x300_ssd_iter_140000.caffemodel")

print(f"Prototxt: {proto_size} bytes (expected ~28KB)")
print(f"Model: {model_size} bytes (expected ~10MB)")

# Re-download if sizes don't match
```

---

### ❌ Error: Window freezes / hangs

**Cause:** Missing `cv2.waitKey()` in the loop.

**Fix:** Always include `cv2.waitKey(1)` inside your loop:
```python
# This is REQUIRED for the window to update
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

---

### ❌ Error: `(-215:Assertion failed)` errors

**Cause:** Empty frame or invalid image passed to OpenCV functions.

**Fix:**
```python
ret, frame = cap.read()
if not ret or frame is None:
    print("Skipping empty frame")
    continue  # Skip instead of crash
```

---

## 3. Debugging Tips

### 3.1 Print Detection Coordinates

```python
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for i, (x, y, w, h) in enumerate(faces):
    print(f"Face {i+1}: x={x}, y={y}, w={w}, h={h}")
```

### 3.2 Display Intermediate Frames

```python
# Show the grayscale frame to verify preprocessing
cv2.imshow("Grayscale", gray)

# Show equalized histogram
equalized = cv2.equalizeHist(gray)
cv2.imshow("Equalized", equalized)
```

### 3.3 Log FPS to Console

```python
fps = fps_counter.update()
if int(fps) % 10 == 0:  # Print every ~10 frames
    print(f"FPS: {fps:.1f} | Faces: {len(faces)}")
```

### 3.4 Save Detection Frame for Analysis

```python
# Save a frame when detection occurs for later analysis
if len(faces) > 0:
    cv2.imwrite(f"debug_frame_{int(time.time())}.jpg", frame)
```

---

## 4. Performance Verification Checklist

| Test                                | Expected Result        | Status |
|-------------------------------------|------------------------|--------|
| Webcam opens successfully           | No errors              | ☐      |
| Haar Cascade model loads            | No assertion errors    | ☐      |
| DNN model loads                     | Model files verified   | ☐      |
| Single face detected                | 1 bounding box drawn   | ☐      |
| Multiple faces detected             | Count matches visually | ☐      |
| FPS counter displayed               | > 15 FPS               | ☐      |
| `q` key exits cleanly               | Window closes, no hang | ☐      |
| No face → count shows 0             | Overlay shows 0        | ☐      |
| Works in low light                  | Detection still fires  | ☐      |

---

## ⏭️ Next Step

👉 Proceed to **[08 — Optimization](./08_optimization.md)**
