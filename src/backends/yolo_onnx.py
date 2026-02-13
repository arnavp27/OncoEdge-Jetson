"""ONNX Runtime backend for YOLO detector.

Supports YOLO models exported to ONNX via `yolo export format=onnx`.
Ultralytics ONNX exports include NMS postprocessing in the graph.
"""
import time

import cv2
import numpy as np
import onnxruntime as ort

from src.interfaces.detector import BaseDetector, Detection, DetectorOutput
from src.backends.registry import register_detector


# NOTE: We use a distinct extension pattern '.onnx' for the classifier,
# so we register this detector under a special key to avoid conflicts.
# In practice, the registry dispatches based on role (detector vs classifier),
# but since both use '.onnx', we need the factory functions
# create_detector() and create_classifier() which use separate registries.


@register_detector('.onnx')
class ONNXYOLODetector(BaseDetector):
    """YOLO detector using ONNX Runtime.

    Expects a YOLO model exported to ONNX via:
        yolo export model=yolo11n-seg.pt format=onnx

    Note: Ultralytics ONNX exports may include or exclude NMS.
    This implementation handles the standard export format.
    """

    def __init__(self, model_path: str, device: str = 'cpu',
                 imgsz: int = 640, **kwargs):
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.session = None

    def load(self) -> None:
        """Load ONNX model."""
        providers = ['CPUExecutionProvider']
        if self.device in ('cuda', '0'):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        print(f"ONNX YOLO detector loaded: {self.model_path}")

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> DetectorOutput:
        """Run YOLO detection via ONNX Runtime.

        Args:
            image: numpy array (H, W, 3) uint8 RGB
            conf_threshold: minimum confidence threshold

        Returns:
            DetectorOutput with standardized detections
        """
        t0 = time.perf_counter()
        orig_h, orig_w = image.shape[:2]

        # Preprocess: resize, normalize, transpose to NCHW
        input_tensor = self._preprocess(image)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})

        detections = self._postprocess(outputs, orig_h, orig_w, conf_threshold)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DetectorOutput(
            detections=detections,
            orig_shape=(orig_h, orig_w),
            inference_time_ms=elapsed_ms,
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO ONNX model."""
        resized = cv2.resize(image, (self.imgsz, self.imgsz))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))    # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)       # NCHW
        return blob

    def _postprocess(self, outputs, orig_h, orig_w, conf_threshold) -> list:
        """Parse ONNX output and convert to Detection list.

        Note: The exact output format depends on the YOLO export settings.
        This is a template implementation that handles the common case
        of ultralytics ONNX exports with post-NMS output.
        """
        detections = []

        # Ultralytics ONNX export typically outputs:
        # outputs[0]: shape (1, num_detections, 4+num_classes+32) for seg models
        # The exact parsing depends on the export version.
        # This template handles the basic bounding-box case.

        if len(outputs) == 0 or outputs[0] is None:
            return detections

        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]  # Remove batch dimension

        # Scale factors from imgsz back to original
        scale_x = orig_w / self.imgsz
        scale_y = orig_h / self.imgsz

        for row in raw:
            if len(row) < 5:
                continue

            # Common format: [x1, y1, x2, y2, conf, class_id, ...]
            # or [cx, cy, w, h, conf, class_scores...]
            # Adapt based on actual YOLO ONNX export format
            conf = float(row[4]) if len(row) > 4 else 0.0
            if conf < conf_threshold:
                continue

            x1 = int(row[0] * scale_x)
            y1 = int(row[1] * scale_y)
            x2 = int(row[2] * scale_x)
            y2 = int(row[3] * scale_y)

            cls_id = int(row[5]) if len(row) > 5 else 0

            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                confidence=conf,
                mask=None,
                class_id=cls_id,
                class_name=str(cls_id),
            ))

        return detections

    def get_model_info(self) -> dict:
        return {
            'format': '.onnx (YOLO)',
            'model_path': self.model_path,
            'device': self.device,
            'imgsz': self.imgsz,
        }
