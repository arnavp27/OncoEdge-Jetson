"""Abstract interface for image classifiers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ClassificationResult:
    """Standardized classification output."""

    predicted_class: str = ""                               # Top predicted class name
    class_scores: Dict[str, float] = field(default_factory=dict)  # class name -> probability
    max_score: float = 0.0                                  # Probability of top class
    embedding: Optional[np.ndarray] = None                  # Raw image embedding (optional)
    inference_time_ms: float = 0.0                          # Latency tracking


class BaseClassifier(ABC):
    """Abstract classifier interface.

    All classifier backends (open_clip .bin, ONNX, TensorRT)
    must implement this interface.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def classify(self, image_patch: np.ndarray) -> ClassificationResult:
        """Classify a single image patch.

        Args:
            image_patch: numpy array (H, W, 3) uint8 RGB, arbitrary size

        Returns:
            ClassificationResult with class probabilities
        """
        ...

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return ordered list of class names."""
        ...

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return metadata about the loaded model."""
        ...
