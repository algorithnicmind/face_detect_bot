# 🤖 DNN-Based Face Detection — Deep Dive

> **Document:** 05 — DNN Detection Guide  
> **Version:** 1.0  
> **Last Updated:** March 23, 2026

---

## 📌 Overview

OpenCV's **DNN module** uses a pre-trained **SSD (Single Shot MultiBox Detector)** with **ResNet-10** for more accurate face detection than Haar Cascades.

---

## 1. DNN vs Haar Cascade

| Feature              | Haar Cascade | DNN (SSD+ResNet) |
|----------------------|-------------|-------------------|
| Accuracy             | ⭐⭐⭐     | ⭐⭐⭐⭐⭐       |
| Speed                | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐          |
| False Positives      | Higher       | Very Low          |
| Tilted Faces         | ❌ Poor      | ✅ Good           |
| Lighting Robustness  | ❌ Sensitive  | ✅ Robust         |

---

## 2. Required Model Files

| File | Description |
|------|-------------|
| `deploy.prototxt` | Network architecture (~28 KB) |
| `res10_300x300_ssd_iter_140000.caffemodel` | Weights (~10 MB) |

### Download:
```bash
curl -O https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -O https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

Place in the `models/` directory.

---

## 3. Step-by-Step Implementation

### Step 1: Load the Model
```python
import cv2, numpy as np

net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)
```

### Step 2: Create Blob (Preprocess)
```python
blob = cv2.dnn.blobFromImage(
    frame, 1.0, (300, 300),
    (104.0, 177.0, 123.0),  # Mean subtraction (BGR)
    swapRB=False, crop=False
)
```

### Step 3: Run Inference
```python
net.setInput(blob)
detections = net.forward()
# Output shape: (1, 1, N, 7) → [batch, class, confidence, x1, y1, x2, y2]
```

### Step 4: Process Detections
```python
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

---

## 4. Complete DNN Detection Code

```python
import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)
THRESHOLD = 0.5
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > THRESHOLD:
            face_count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence*100:.1f}%", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('DNN Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 5. Confidence Threshold Tuning

| Threshold | Behavior |
|-----------|----------|
| `0.3` | More detections, some false positives |
| `0.5` | **Recommended** — good balance |
| `0.7` | High-confidence only |
| `0.9` | Very strict, may miss faces |

---

## 6. GPU Acceleration (Optional)

```python
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

> Requires OpenCV compiled with CUDA support.

---

## ⏭️ Next Step

👉 Proceed to **[06 — Implementation Guide](./06_implementation.md)**
