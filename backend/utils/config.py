import os

# Base directory is the 'backend' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx")

# Database path
DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'database', 'surveillance_production.db')}"

# Configuration thresholds
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
DOWNSCALE_FACTOR = 0.5
TRACKER_MAX_AGE = 30
