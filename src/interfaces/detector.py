"""Abstract interface for object detectors."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Detection:
    """Standardized detection output, decoupled from any ML framework."""

    bbox: List[int]                       # [x1, y1, x2, y2] pixel coordinates
    confidence: float                     # Detection confidence 0-1
    mask: Optional[np.ndarray] = None     # Segmentation mask (H, W) float32, or None
    class_id: int = 0                     # Class ID from detector
    class_name: str = "object"            # Class name string


@dataclass
class DetectorOutput:
    """Collection of detections for a single image."""

    detections: List[Detection] = field(default_factory=list)
    orig_shape: tuple = (0, 0)            # (H, W) of input image
    inference_time_ms: float = 0.0        # Latency tracking


class BaseDetector(ABC):
    """Abstract detector interface.

    All detector backends (ultralytics .pt, ONNX, TensorRT)
    must implement this interface.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> DetectorOutput:
        """Run detection on a single image.

        Args:
            image: numpy array (H, W, 3) uint8 RGB
            conf_threshold: minimum confidence threshold

        Returns:
            DetectorOutput with standardized detections
        """
        ...

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return metadata about the loaded model (format, size, device)."""
        ...
