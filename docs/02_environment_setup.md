# ⚙️ Environment Setup — Real-Time Face Detection System

> **Document:** 02 — Environment Setup  
> **Version:** 1.0  
> **Last Updated:** March 23, 2026  
> **Author:** Ankit (AlgorithmicMind)

---

## 📌 Overview

This document walks you through setting up your development environment from scratch — installing Python, creating a virtual environment, and installing all required dependencies.

---

## Step 1: Verify Python Installation

Open your terminal (Command Prompt / PowerShell / Terminal) and run:

```bash
python --version
```

**Expected Output:**
```
Python 3.10.x   (or any version >= 3.8)
```

> ⚠️ If Python is not installed, download it from [python.org](https://www.python.org/downloads/) and make sure to check **"Add Python to PATH"** during installation.

---

## Step 2: Verify pip Installation

```bash
pip --version
```

**Expected Output:**
```
pip 23.x.x from ... (python 3.10)
```

> If pip is not available, run:
> ```bash
> python -m ensurepip --upgrade
> ```

---

## Step 3: Clone the Repository

```bash
git clone https://github.com/algorithmicmind/face_detect_bot.git
cd face_detect_bot
```

---

## Step 4: Create a Virtual Environment

A virtual environment isolates project dependencies from your system Python.

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**After activation, your prompt will show:**
```
(venv) C:\Users\ankit\...\face_detect_bot>
```

---

## Step 5: Install Dependencies

### Option A — Using requirements.txt (Recommended)

Create a `requirements.txt` file in the project root with the following contents:

```txt
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80
numpy==1.26.4
```

Then install:

```bash
pip install -r requirements.txt
```

### Option B — Manual Installation

```bash
pip install opencv-python opencv-contrib-python numpy
```

---

## Step 6: Verify OpenCV Installation

Run the following in Python to confirm OpenCV is installed:

```bash
python -c "import cv2; print(f'OpenCV Version: {cv2.__version__}')"
```

**Expected Output:**
```
OpenCV Version: 4.9.0
```

---

## Step 7: Test Webcam Access

Run this quick test to verify your webcam is accessible:

```python
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot access webcam")
else:
    print("✅ Webcam is working!")
    ret, frame = cap.read()
    if ret:
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()
```

Save this as `test_webcam.py` and run:

```bash
python test_webcam.py
```

**Expected Output:**
```
✅ Webcam is working!
   Frame size: 640x480
```

---

## Step 8: Download Pre-trained Models

### Haar Cascade (Built-in with OpenCV)

The Haar Cascade XML file comes bundled with OpenCV. To find its location:

```python
import cv2
import os

haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
print(f"Haar Cascade path: {haar_path}")
print(f"File exists: {os.path.exists(haar_path)}")
```

You can also copy it to your project's `models/` directory:

```bash
mkdir models
```

Then copy the file from OpenCV's data directory into `models/`.

### DNN Model (Optional — for better accuracy)

Download the following two files for the DNN-based detector:

| File | URL |
|------|-----|
| `deploy.prototxt` | [GitHub - opencv/samples](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt) |
| `res10_300x300_ssd_iter_140000.caffemodel` | [GitHub - opencv_3rdparty](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel) |

Place both files in the `models/` folder.

---

## 📁 Environment Checklist

| #  | Step                                | Status |
|----|-------------------------------------|--------|
| 1  | Python 3.8+ installed               | ✅ Done |
| 2  | pip available                       | ✅ Done |
| 3  | Repository cloned                   | ✅ Done |
| 4  | Virtual environment created         | ✅ Done |
| 5  | Dependencies installed              | ✅ Done |
| 6  | OpenCV verified (4.13.0)            | ✅ Done |
| 7  | Webcam tested                       | ✅ Done |
| 8  | Model files downloaded              | ✅ Done |

> ✅ **Environment setup complete!** All 8 steps finished on March 23, 2026.

---

## 🔧 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'cv2'`
- Ensure your virtual environment is activated
- Run `pip install opencv-python` again

### ❌ `Cannot access webcam`
- Check if another application is using the webcam
- Try `cv2.VideoCapture(1)` for external USB cameras
- On Linux, ensure you have permissions: `sudo chmod 666 /dev/video0`

### ❌ `pip install fails with permission error`
- Use `pip install --user opencv-python`
- Or run terminal as Administrator (Windows)

---

## ⏭️ Next Step

👉 Proceed to **[03 — Understanding OpenCV](./03_understanding_opencv.md)** to learn the core OpenCV concepts used in this project.
