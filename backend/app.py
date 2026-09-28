import logging
import io
from typing import List, Dict, Any, Optional
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import our custom modules
from detection.motion import MotionDetector
from detection.object import ObjectDetector
from detection.tracking import ObjectTracker
from database.db_utils import DatabaseManager
from utils.config import DB_PATH, YOLO_MODEL_PATH, DOWNSCALE_FACTOR, TRACKER_MAX_AGE, CONFIDENCE_THRESHOLD
from utils.helpers import get_class_name, is_suspicious

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Service Layer Globals ---
# In a large production app, use Dependency Injection. For this prototype, globals suffice.
db: DatabaseManager
motion_detector: MotionDetector
object_detector: ObjectDetector
tracker: ObjectTracker

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Manager.
    Handles startup (loading heavy AI models, connecting to DB) and shutdown gracefully.
    """
    global db, motion_detector, object_detector, tracker
    logger.info("Initializing API services and loading AI models...")
    
    try:
        # Initialize Database
        db = DatabaseManager(DB_PATH)
        db.run_migrations()
        
        # Initialize Pipeline Components
        motion_detector = MotionDetector(downscale_factor=DOWNSCALE_FACTOR)
        
        # We wrap AI model loading in try-except so the API still boots if the ONNX file is missing
        try:
            object_detector = ObjectDetector(model_path=YOLO_MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD)
            logger.info("ObjectDetector loaded successfully.")
        except Exception as e:
            logger.error(f"ObjectDetector failed to load. Ensure model exists at {YOLO_MODEL_PATH}. Error: {e}")
            object_detector = None
            
        try:
            tracker = ObjectTracker(embedder="mobilenet", max_age=TRACKER_MAX_AGE)
            logger.info("ObjectTracker loaded successfully.")
        except Exception as e:
            logger.error(f"ObjectTracker failed to load. Error: {e}")
            tracker = None

        logger.info("All startup services initialized successfully.")
        yield
    finally:
        logger.info("Shutting down API and releasing resources...")


# --- FastAPI Application ---
app = FastAPI(
    title="Intelligent Surveillance REST API",
    description="Real-time backend for video analytics, motion detection, and object tracking.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Data Models (Schemas) ---

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectionResult(BaseModel):
    class_id: int
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None

class DetectResponse(BaseModel):
    motion_detected: bool
    detections: List[DetectionResult]
    alerts_triggered: int
    processing_time_ms: float

class EventResponse(BaseModel):
    id: int
    timestamp: datetime
    camera_id: str
    object_type: str
    confidence: float
    bounding_box: List[int]

class AlertResponse(BaseModel):
    alert_id: str
    event: EventResponse
    message: str
    severity: str

# --- API Router ---
router = APIRouter()


@router.post("/detect", response_model=DetectResponse, tags=["Inference Pipeline"])
async def process_frame(camera_id: str = Form("CAM_FRONT"), file: UploadFile = File(...)):
    """
    **Primary Inference Endpoint:**
    1. Accepts an uploaded video frame.
    2. Runs fast Motion Detection (MOG2).
    3. If motion exists, runs YOLOv8 Object Detection.
    4. Tracks objects using DeepSORT.
    5. Logs 'suspicious' events (like people/vehicles) to the database.
    """
    start_time = datetime.now()
    
    # 1. Read and decode image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Empty or invalid image data.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    motion_detected = False
    final_detections = []
    alerts_triggered = 0

    try:
        # 2. Motion Detection Phase (Wake-up filter)
        if motion_detector:
            motion_detected, _ = motion_detector.detect_motion(frame)
        else:
            motion_detected = True # Fallback if motion detector is disabled

        # 3. Object Detection & Tracking Phase (Heavy AI)
        if motion_detected and object_detector:
            # Run YOLO
            raw_detections = object_detector.detect_objects(frame)
            
            # Run DeepSORT Tracking
            tracked_objects = []
            if tracker and raw_detections:
                tracked_objects = tracker.track_objects(raw_detections, frame)
            
            # Standardize outputs based on what models are active
            valid_objects = tracked_objects if tracker and tracked_objects else raw_detections

            for obj in valid_objects:
                cls_id = obj.get("class_id", -1)
                # Trackers often drop confidence to prioritize track ID, default to 1.0 if missing
                conf = obj.get("confidence", 1.0) 
                bbox = obj["bbox"]
                t_id = obj.get("track_id")
                
                # Format for JSON response
                res = DetectionResult(
                    class_id=cls_id,
                    confidence=conf,
                    bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
                    track_id=t_id
                )
                final_detections.append(res)
                
                # 4. Analytics & Event Logging Phase
                class_name = get_class_name(cls_id)
                if is_suspicious(class_name):
                    # Trigger alert count
                    alerts_triggered += 1
                    # Log persistently to Database
                    if db:
                        db.log_event(
                            camera_id=camera_id, 
                            object_type=class_name, 
                            confidence=conf, 
                            bounding_box=bbox
                        )

    except Exception as e:
        logger.error(f"Pipeline error during frame processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during AI processing.")

    # Calculate latency
    process_time = (datetime.now() - start_time).total_seconds() * 1000

    return DetectResponse(
        motion_detected=motion_detected,
        detections=final_detections,
        alerts_triggered=alerts_triggered,
        processing_time_ms=process_time
    )


@router.get("/events", response_model=List[EventResponse], tags=["Data & Analytics"])
async def fetch_events(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    object_type: Optional[str] = Query(None, description="Filter by object class (e.g. 'person')"),
    min_confidence: Optional[float] = Query(None, description="Minimum confidence score"),
    limit: int = Query(100, le=1000, description="Max records to return")
):
    """
    **Query Event Logs:**
    Fetches raw historical detections stored in the database. Supports dynamic filtering.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database service is offline.")
    
    try:
        events = db.get_events(
            camera_id=camera_id, 
            object_type=object_type, 
            min_confidence=min_confidence, 
            limit=limit
        )
        return events
    except Exception as e:
        logger.error(f"Database fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve events.")


@router.get("/alerts", response_model=List[AlertResponse], tags=["Data & Analytics"])
async def fetch_alerts(limit: int = Query(50, description="Number of recent alerts to fetch")):
    """
    **Query High-Priority Alerts:**
    Filters the event log specifically for high-priority security breaches (e.g., unauthorized persons).
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database service is offline.")
    
    try:
        # Example rule: an 'alert' is any time a person is detected
        events = db.get_events(object_type="person", limit=limit)
        
        alerts = []
        for event in events:
            alerts.append(
                AlertResponse(
                    alert_id=f"ALT-{event['id']}-{event['camera_id']}",
                    event=event,
                    message=f"Suspicious {event['object_type']} detected on camera {event['camera_id']}",
                    severity="HIGH"
                )
            )
        return alerts
    except Exception as e:
        logger.error(f"Alert fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts.")

# Mount the router
app.include_router(router, prefix="/api/v1")

# Allow running directly via python app.py
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
