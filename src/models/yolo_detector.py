"""
YOLO26n-seg Detector Wrapper

Provides a clean interface for oral lesion detection and segmentation.
"""
from ultralytics import YOLO
import numpy as np
from PIL import Image


class YOLODetector:
    """
    Wrapper for YOLO26n-seg model for lesion detection and segmentation.

    This class handles loading the YOLO model and running inference on images
    to detect and segment oral lesions.
    """

    def __init__(self, model_path='yolo26n-seg.pt', device='0'):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model weights file (default: 'yolo26n-seg.pt')
            device: Device to run inference on ('0' for GPU, 'cpu' for CPU)
        """
        self.model_path = model_path
        self.device = device

        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

    def detect(self, image, conf_threshold=0.25, imgsz=640):
        """
        Run detection on an image.

        Args:
            image: Input image (numpy array HxWx3 or PIL Image)
            conf_threshold: Confidence threshold for detections (0-1)
            imgsz: Input image size for YOLO (will be resized)

        Returns:
            ultralytics Results object containing:
                - boxes: Bounding boxes in xyxy format
                - masks: Segmentation masks (if available)
                - conf: Confidence scores
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Run YOLO inference
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            save=False
        )

        # Return first result (single image inference)
        return results[0]

    def get_detection_count(self, results):
        """
        Get number of detections from results.

        Args:
            results: ultralytics Results object

        Returns:
            int: Number of detections
        """
        if results.boxes is None:
            return 0
        return len(results.boxes)

    def has_masks(self, results):
        """
        Check if results contain segmentation masks.

        Args:
            results: ultralytics Results object

        Returns:
            bool: True if masks are available
        """
        return results.masks is not None and len(results.masks) > 0
