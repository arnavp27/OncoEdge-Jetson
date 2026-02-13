"""Ultralytics .pt backend for YOLO detector."""
import time

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.interfaces.detector import BaseDetector, Detection, DetectorOutput
from src.backends.registry import register_detector


@register_detector('.pt')
class UltralyticsDetector(BaseDetector):
    """YOLO detector using ultralytics .pt format.

    Wraps the ultralytics YOLO class and converts outputs to
    standardized Detection/DetectorOutput dataclasses.
    """

    def __init__(self, model_path: str = 'yolo11n-seg.pt',
                 device: str = '0', imgsz: int = 640, **kwargs):
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.model = None

    def load(self) -> None:
        """Load YOLO model from .pt file."""
        self.model = YOLO(self.model_path)
        print(f"YOLO model loaded successfully from {self.model_path}")

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> DetectorOutput:
        """Run YOLO detection and return standardized output.

        Args:
            image: numpy array (H, W, 3) uint8 RGB
            conf_threshold: minimum confidence threshold

        Returns:
            DetectorOutput with Detection dataclasses
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            save=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = results[0]
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i].cpu().numpy()
                conf = float(result.boxes.conf[i].cpu().numpy())
                cls_id = int(result.boxes.cls[i].cpu().numpy()) if result.boxes.cls is not None else 0
                cls_name = result.names.get(cls_id, str(cls_id)) if hasattr(result, 'names') else str(cls_id)

                mask = None
                if result.masks is not None and i < len(result.masks):
                    mask = result.masks.data[i].cpu().numpy()

                detections.append(Detection(
                    bbox=list(map(int, box)),
                    confidence=conf,
                    mask=mask,
                    class_id=cls_id,
                    class_name=cls_name,
                ))

        return DetectorOutput(
            detections=detections,
            orig_shape=image.shape[:2],
            inference_time_ms=elapsed_ms,
        )

    def get_model_info(self) -> dict:
        return {
            'format': '.pt (ultralytics)',
            'model_path': self.model_path,
            'device': self.device,
            'imgsz': self.imgsz,
        }
