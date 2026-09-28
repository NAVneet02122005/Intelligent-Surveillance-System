import logging
from typing import List, Dict, Any

import numpy as np

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    logger.warning("The 'deep-sort-realtime' package is required. Please install it using: pip install deep-sort-realtime")
    DeepSort = None


class TrackingError(Exception):
    """Custom exception for object tracking errors."""
    pass


class ObjectTracker:
    """
    Multi-Object Tracker utilizing DeepSORT.
    
    Maintains object identities across frames, handling occlusions and
    re-identifications using appearance embeddings and Kalman filtering.
    A clean separation is maintained between the detection phase and tracking phase.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3, embedder: str = "mobilenet", max_cosine_distance: float = 0.2):
        """
        Initializes the DeepSORT tracker.

        Args:
            max_age (int): Maximum number of missed frames before a track is deleted.
            n_init (int): Number of consecutive detections before the track is confirmed.
            embedder (str): Name of the feature extraction model used for ReID (e.g., 'mobilenet').
            max_cosine_distance (float): Gating threshold for cosine distance metric matching.
        """
        if DeepSort is None:
            logger.error("deep_sort_realtime library is missing.")
            raise ImportError("deep_sort_realtime library is missing. Run: pip install deep-sort-realtime")

        try:
            # Initialize DeepSort with the chosen parameters.
            # nms_max_overlap is set to 1.0 because Non-Maximum Suppression (NMS) 
            # should already be handled during the object detection phase.
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_cosine_distance=max_cosine_distance,
                embedder=embedder,
                nms_max_overlap=1.0
            )
            logger.info(f"DeepSORT initialized successfully with embedder: {embedder}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSORT: {str(e)}")
            raise TrackingError(f"Tracker initialization failed: {str(e)}")

    def track_objects(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Updates tracking states given new detections and the current video frame.
        Extracts appearance embeddings from the frame to associate detections with existing tracks.

        Args:
            detections (List[Dict[str, Any]]): Detections from the detection module.
                                               Format: [{"class_id": int, "confidence": float, "bbox": [x1, y1, x2, y2]}]
            frame (np.ndarray): The current BGR video frame used for extracting appearance features.

        Returns:
            List[Dict[str, Any]]: Tracked objects with persistent unique IDs.
                                  Format: [{"track_id": int, "class_id": int, "bbox": [x1, y1, x2, y2]}]
        """
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid input frame provided to tracker.")
            raise TrackingError("Input frame must be a valid numpy.ndarray.")

        # If there are no detections, we still need to update the tracker so it can 
        # increment the 'missed' age of existing tracks (for the Kalman filter).
        if not detections:
            self.tracker.update_tracks([], frame=frame)
            return []

        try:
            # 1. Convert detections to DeepSORT format: ([left, top, w, h], confidence, detection_class)
            formatted_detections = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                w = x2 - x1
                h = y2 - y1
                
                # Sanity check to prevent invalid bounding boxes
                if w > 0 and h > 0:
                    formatted_detections.append(
                        ([x1, y1, w, h], det["confidence"], det["class_id"])
                    )

            # 2. Update tracks via DeepSORT engine
            tracks = self.tracker.update_tracks(formatted_detections, frame=frame)

            # 3. Format the tracked outputs for the standardized API
            tracked_objects = []
            for track in tracks:
                # Only return tracks that are confirmed (seen n_init times) 
                # and have been updated in the current frame
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                # Extract bounding box in [left, top, right, bottom] format
                ltrb = track.to_ltrb() 
                track_id = int(track.track_id)
                
                # Extract class ID safely
                try:
                    class_id = track.get_det_class()
                except AttributeError:
                    class_id = getattr(track, 'det_class', -1)
                    
                class_id = int(class_id) if class_id is not None else -1

                tracked_objects.append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "bbox": [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
                })

            return tracked_objects

        except Exception as e:
            logger.error(f"Error during tracking update: {str(e)}")
            raise TrackingError(f"Tracking update failed: {str(e)}")

# Example usage (commented out):
# if __name__ == "__main__":
#     import cv2
#     
#     # Initialize tracker
#     tracker = ObjectTracker(max_age=30, embedder="mobilenet")
#     
#     # Mock frame and detections (from object.py)
#     frame = np.zeros((480, 640, 3), dtype=np.uint8)
#     detections = [
#         {"class_id": 0, "confidence": 0.9, "bbox": [100, 100, 200, 300]}
#     ]
#     
#     # Run tracking
#     tracked_objects = tracker.track_objects(detections, frame)
#     print(tracked_objects)
