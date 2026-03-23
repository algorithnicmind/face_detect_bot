# 📦 Deployment & Packaging Guide

> **Document:** 09 — Deployment Guide  
> **Version:** 1.0  
> **Last Updated:** March 23, 2026  
> **Author:** Ankit (AlgorithmicMind)

---

## 📌 Overview

This document covers how to package, distribute, and deploy the Face Detection System — from creating executables to running on different platforms.

---

## 1. Creating a Standalone Executable

### Using PyInstaller

PyInstaller bundles your Python app into a single `.exe` (Windows) or binary.

#### Install PyInstaller:
```bash
pip install pyinstaller
```

#### Build the executable:
```bash
# Single-file executable
pyinstaller --onefile --windowed src/face_detector_haar.py

# With model files included
pyinstaller --onefile --windowed \
    --add-data "models;models" \
    src/face_detector_haar.py
```

#### Flags explained:

| Flag | Purpose |
|------|---------|
| `--onefile` | Bundle everything into a single `.exe` |
| `--windowed` | No console window (GUI only) |
| `--add-data` | Include non-Python files (models) |
| `--icon=icon.ico` | Custom application icon |
| `--name=FaceDetector` | Custom output filename |

#### Output:
```
face_detect_bot/
├── dist/
│   └── face_detector_haar.exe   ← Distributable executable
├── build/                        ← Build artifacts (can delete)
└── face_detector_haar.spec       ← Build configuration
```

---

## 2. Creating a `.spec` File for Custom Builds

For more control, create a `face_detector.spec`:

```python
# face_detector.spec
a = Analysis(
    ['src/face_detector_haar.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models/*.xml', 'models'),
        ('models/*.prototxt', 'models'),
        ('models/*.caffemodel', 'models'),
    ],
    hiddenimports=['cv2', 'numpy'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FaceDetector',
    debug=False,
    strip=False,
    upx=True,
    console=False,
)
```

Build with:
```bash
pyinstaller face_detector.spec
```

---

## 3. Docker Deployment (Linux / Cloud)

### Dockerfile:

```dockerfile
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

# Note: Webcam access in Docker requires --device flag
CMD ["python", "src/face_detector_haar.py"]
```

### Build & Run:

```bash
# Build the image
docker build -t face-detector .

# Run with webcam access (Linux only)
docker run --device=/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    face-detector
```

> ⚠️ Docker webcam access works natively on Linux. On Windows/macOS, use a virtual display or stream the video over a network socket.

---

## 4. Cross-Platform Notes

### Windows
- ✅ `cv2.VideoCapture(0)` works out of the box
- ✅ PyInstaller creates `.exe` files
- Use `cv2.CAP_DSHOW` for DirectShow backend:
  ```python
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  ```

### macOS
- ⚠️ Requires camera access permission (System Settings → Privacy)
- Use `cv2.CAP_AVFOUNDATION` backend:
  ```python
  cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
  ```

### Linux
- ⚠️ May need `v4l2` utilities: `sudo apt install v4l-utils`
- Check available cameras: `v4l2-ctl --list-devices`
- Grant permissions: `sudo chmod 666 /dev/video0`

---

## 5. Creating a `.gitignore`

Essential `.gitignore` for this project:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.egg-info/
dist/
build/
*.spec

# Virtual environment
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp

# OS files
.DS_Store
Thumbs.db

# Large model files (optional — if not tracking in git)
# models/*.caffemodel

# Debug output
debug_frame_*.jpg
```

---

## 6. README.md Template

Create a professional `README.md` in the project root:

```markdown
# 🎯 Real-Time Face Detection System

Detect human faces in live webcam video using OpenCV.

## ✨ Features
- Real-time face detection via webcam
- Bounding rectangles around detected faces
- Live face count display
- FPS counter overlay
- Haar Cascade & DNN dual implementation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam

### Installation
git clone https://github.com/algorithmicmind/face_detect_bot.git
cd face_detect_bot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### Run
# Haar Cascade version
python src/face_detector_haar.py

# DNN version (download models first)
python src/face_detector_dnn.py

## 📖 Documentation
See the [docs/](./docs/) folder for detailed guides.

## 🛠️ Tech Stack
- Python 3.10
- OpenCV 4.9
- NumPy

## 📄 License
Apache 2.0
```

---

## 7. GitHub Repository Setup

### Initialize & push:
```bash
git init
git add .
git commit -m "Initial commit: Real-Time Face Detection System"
git branch -M main
git remote add origin https://github.com/algorithmicmind/face_detect_bot.git
git push -u origin main
```

### Recommended branch strategy:
```
main           ← Stable releases
├── develop    ← Active development
├── feature/*  ← New features
└── bugfix/*   ← Bug fixes
```

---

## 8. Deployment Checklist

| # | Task | Status |
|---|------|--------|
| 1 | All source files created (`src/`) | ☐ |
| 2 | Model files in `models/` | ☐ |
| 3 | `requirements.txt` complete | ☐ |
| 4 | `.gitignore` configured | ☐ |
| 5 | `README.md` written | ☐ |
| 6 | Documentation in `docs/` | ☐ |
| 7 | Tested on target platform | ☐ |
| 8 | Executable built (optional) | ☐ |
| 9 | Pushed to GitHub | ☐ |

---

## 📚 Full Documentation Index

| # | Document | Topic |
|---|----------|-------|
| 01 | [Project Overview](./01_project_overview.md) | Summary, features, architecture |
| 02 | [Environment Setup](./02_environment_setup.md) | Installation & configuration |
| 03 | [Understanding OpenCV](./03_understanding_opencv.md) | Core OpenCV concepts |
| 04 | [Haar Cascade Guide](./04_haar_cascade_guide.md) | Haar algorithm deep dive |
| 05 | [DNN Detection Guide](./05_dnn_detection_guide.md) | DNN-based detection |
| 06 | [Implementation](./06_implementation.md) | Complete code walkthrough |
| 07 | [Testing & Debugging](./07_testing_debugging.md) | Testing & troubleshooting |
| 08 | [Optimization](./08_optimization.md) | Performance tuning |
| 09 | [Deployment Guide](./09_deployment_guide.md) | Packaging & deployment |

---

> 🎉 **Congratulations!** You've completed the full documentation for the Real-Time Face Detection System.
