import os
import logging
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


class ObjectDetectionError(Exception):
    """Custom exception for object detection errors."""
    pass


class ObjectDetector:
    """
    YOLOv8 ONNX Runtime Object Detector.
    
    Loads an exported YOLOv8 ONNX model and performs inference with optimized
    preprocessing and postprocessing, including Non-Maximum Suppression (NMS).
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45, providers: List[str] = None):
        """
        Initializes the ObjectDetector.

        Args:
            model_path (str): Path to the YOLOv8 ONNX model file.
            conf_threshold (float): Confidence threshold for detections (0.0 to 1.0).
            iou_threshold (float): Intersection over Union threshold for NMS.
            providers (List[str], optional): ONNX Runtime execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        if providers is None:
            # Default to CPU if no providers specified. If ONNX Runtime GPU is installed, CUDA is usually preferred.
            providers = ['CPUExecutionProvider']

        try:
            logger.info(f"Loading ONNX model from {model_path} with providers: {providers}")
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Get model input details
            model_inputs = self.session.get_inputs()
            self.input_name = model_inputs[0].name
            self.input_shape = model_inputs[0].shape
            
            # YOLOv8 input shape is typically (1, 3, H, W)
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX Runtime session: {str(e)}")
            raise ObjectDetectionError(f"Failed to load model: {str(e)}")

    def _letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Resizes image to a square while preserving the aspect ratio, padding with a solid color.
        """
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Calculate unpadded dimensions
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding widths
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Prepares the frame for ONNX inference.
        """
        # Resize with padding
        img, ratio, dwdh = self._letterbox(frame, new_shape=(self.input_height, self.input_width))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # HWC to CHW format: (H, W, 3) -> (3, H, W)
        img = img.transpose((2, 0, 1))
        
        # Expand dimensions: (3, H, W) -> (1, 3, H, W)
        img = np.expand_dims(img, 0)
        
        # Normalize to [0, 1] and convert to contiguous array
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        
        return img, ratio, dwdh

    def _postprocess(self, outputs: List[np.ndarray], ratio: float, dwdh: Tuple[float, float], orig_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Processes YOLOv8 raw ONNX outputs and applies Non-Maximum Suppression (NMS).
        """
        # YOLOv8 output shape is typically (1, 84, 8400) where 84 = 4 (bbox) + 80 (classes)
        predictions = np.squeeze(outputs[0]).T  # Transpose to (8400, 84)
        
        if len(predictions) == 0:
            return []

        # Get max score and corresponding class for each bounding box
        scores = np.max(predictions[:, 4:], axis=1)
        
        # Filter out low-confidence predictions early to speed up processing
        mask = scores > self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        
        if len(predictions) == 0:
            return []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]

        # Convert bbox format from [cx, cy, w, h] to [x1, y1, x2, y2]
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Rescale boxes back to the original image dimensions
        boxes[:, 0] = (boxes[:, 0] - dwdh[0]) / ratio
        boxes[:, 1] = (boxes[:, 1] - dwdh[1]) / ratio
        boxes[:, 2] = (boxes[:, 2] - dwdh[0]) / ratio
        boxes[:, 3] = (boxes[:, 3] - dwdh[1]) / ratio

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices:
                # Handle different return types of NMSBoxes depending on OpenCV version
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                box = boxes[idx].astype(int).tolist()
                
                # Ensure bounding box coordinates stay within the image boundaries
                x1, y1, x2, y2 = box
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_shape[1], x2)
                y2 = min(orig_shape[0], y2)

                results.append({
                    "class_id": int(class_ids[idx]),
                    "confidence": float(scores[idx]),
                    "bbox": [x1, y1, x2, y2]
                })

        return results

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects objects in a given video frame.

        Args:
            frame (np.ndarray): Input video frame in BGR format.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing detection results.
                                  Format: [{"class_id": int, "confidence": float, "bbox": [x1, y1, x2, y2]}, ...]
        """
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid input frame provided.")
            raise ObjectDetectionError("Input frame must be a valid numpy.ndarray.")
        
        try:
            orig_shape = frame.shape[:2]  # (height, width)
            
            # Preprocess
            input_tensor, ratio, dwdh = self._preprocess(frame)
            
            # Inference
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            # Postprocess
            detections = self._postprocess(outputs, ratio, dwdh, orig_shape)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            raise ObjectDetectionError(f"Detection failed: {str(e)}")

# Example usage (commented out):
# if __name__ == "__main__":
#     # Initialize detector
#     detector = ObjectDetector(model_path="yolov8n.onnx", conf_threshold=0.5)
#     
#     # Read frame
#     frame = cv2.imread("test.jpg")
#     
#     # Run detection
#     detections = detector.detect_objects(frame)
#     print(detections)
