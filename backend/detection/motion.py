import logging
from typing import List, Tuple

import cv2
import numpy as np

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


class MotionDetectionError(Exception):
    """Custom exception for motion detection errors."""
    pass


class MotionDetector:
    """
    Real-Time Motion Detector using Background Subtraction.
    
    Implements OpenCV's MOG2 or KNN background subtraction algorithms.
    Optimized for speed by downscaling frames and disabling shadow detection,
    making it ideal as a lightweight pre-filter before heavy AI object detection.
    """

    def __init__(self, method: str = 'MOG2', min_area: int = 1000, downscale_factor: float = 0.5):
        """
        Initializes the MotionDetector.

        Args:
            method (str): Background subtraction method to use ('MOG2' or 'KNN'). Default is 'MOG2'.
            min_area (int): Minimum contour area in the original frame size to be considered valid motion.
                            Filters out small noise like leaves or rain.
            downscale_factor (float): Factor to scale down the image before processing to drastically
                                      increase FPS. Bounding boxes are scaled back up automatically.
        """
        self.min_area = min_area
        self.downscale_factor = downscale_factor

        method = method.upper()
        try:
            if method == 'MOG2':
                # history=500, varThreshold=16 are good defaults for surveillance.
                # detectShadows=False significantly speeds up processing.
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=False
                )
            elif method == 'KNN':
                self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                    history=500, dist2Threshold=400.0, detectShadows=False
                )
            else:
                logger.error(f"Unsupported background subtraction method: {method}")
                raise ValueError("Method must be either 'MOG2' or 'KNN'.")
                
            logger.info(f"Initialized MotionDetector with method: {method}, downscale: {downscale_factor}")
        except Exception as e:
            logger.error(f"Failed to initialize background subtractor: {str(e)}")
            raise MotionDetectionError(f"Initialization failed: {str(e)}")

        # Morphological kernel for noise reduction
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, List[List[int]]]:
        """
        Detects motion in the provided video frame.

        Args:
            frame (np.ndarray): Input video frame in BGR format.

        Returns:
            Tuple[bool, List[List[int]]]: 
                - A boolean flag indicating if any motion was detected.
                - A list of bounding boxes for the motion regions. Format: [[x1, y1, x2, y2], ...]
        """
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid input frame provided for motion detection.")
            raise MotionDetectionError("Input frame must be a valid numpy.ndarray.")

        try:
            # 1. Downscale for real-time optimization
            if self.downscale_factor != 1.0:
                working_frame = cv2.resize(
                    frame, 
                    (0, 0), 
                    fx=self.downscale_factor, 
                    fy=self.downscale_factor, 
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                working_frame = frame

            # 2. Convert to Grayscale & Blur to remove high-frequency noise
            gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # 3. Apply Background Subtraction to get the foreground mask
            fg_mask = self.bg_subtractor.apply(gray)

            # 4. Morphological operations to clean up the mask
            # Erode removes tiny specs of noise, Dilate merges disconnected parts of a moving object
            fg_mask = cv2.erode(fg_mask, self.kernel, iterations=1)
            fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

            # 5. Find contours of the moving blobs
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            
            # Area threshold adjusted for the downscaled image
            scaled_min_area = self.min_area * (self.downscale_factor ** 2)

            for contour in contours:
                # 6. Filter by minimum area
                if cv2.contourArea(contour) < scaled_min_area:
                    continue

                # 7. Get bounding box and scale it back to original resolution
                x, y, w, h = cv2.boundingRect(contour)
                
                orig_x1 = int(x / self.downscale_factor)
                orig_y1 = int(y / self.downscale_factor)
                orig_x2 = int((x + w) / self.downscale_factor)
                orig_y2 = int((y + h) / self.downscale_factor)

                regions.append([orig_x1, orig_y1, orig_x2, orig_y2])

            motion_detected = len(regions) > 0
            
            return motion_detected, regions

        except Exception as e:
            logger.error(f"Error during motion detection: {str(e)}")
            raise MotionDetectionError(f"Motion detection failed: {str(e)}")

# Example usage (commented out):
# if __name__ == "__main__":
#     detector = MotionDetector(method='MOG2', min_area=1000)
#     cap = cv2.VideoCapture(0)
#     
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#         
#         has_motion, boxes = detector.detect_motion(frame)
#         
#         if has_motion:
#             for box in boxes:
#                 cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#                 
#         cv2.imshow('Motion Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
