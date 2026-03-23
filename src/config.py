"""
config.py - Centralized configuration for Face Detection System
"""

# Camera Settings
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480

# Haar Cascade Settings
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (30, 30)
HAAR_MAX_SIZE = None  # No upper limit

# DNN Settings
DNN_CONFIDENCE_THRESHOLD = 0.5
DNN_INPUT_SIZE = (300, 300)
DNN_MEAN_SUBTRACTION = (104.0, 177.0, 123.0)

# Display Settings
INFO_OVERLAY_HEIGHT = 50
INFO_OVERLAY_OPACITY = 0.6

# Performance Settings
USE_THREADED_CAPTURE = True
CAPTURE_QUEUE_SIZE = 2
SKIP_FRAMES = 0  # 0 = no skip, 1 = every other frame, 2 = every 3rd frame

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model Paths
MODELS_DIR = "models"
HAAR_MODEL_NAME = "haarcascade_frontalface_default.xml"
DNN_PROTOTXT = "deploy.prototxt"
DNN_CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"
