"""Unit tests for YOLO detector via BaseDetector interface."""
import pytest
import numpy as np

from src.interfaces.detector import Detection, DetectorOutput, BaseDetector


def test_detection_dataclass():
    """Test Detection dataclass creation."""
    det = Detection(
        bbox=[10, 20, 100, 200],
        confidence=0.85,
        mask=np.zeros((640, 640), dtype=np.float32),
        class_id=0,
        class_name="lesion",
    )
    assert det.bbox == [10, 20, 100, 200]
    assert det.confidence == 0.85
    assert det.mask is not None
    assert det.class_id == 0
    assert det.class_name == "lesion"


def test_detection_defaults():
    """Test Detection dataclass defaults."""
    det = Detection(bbox=[0, 0, 50, 50], confidence=0.5)
    assert det.mask is None
    assert det.class_id == 0
    assert det.class_name == "object"


def test_detector_output_dataclass():
    """Test DetectorOutput creation with detections."""
    dets = [
        Detection(bbox=[10, 20, 100, 200], confidence=0.8),
        Detection(bbox=[50, 60, 150, 250], confidence=0.6),
    ]
    output = DetectorOutput(
        detections=dets,
        orig_shape=(480, 640),
        inference_time_ms=45.2,
    )
    assert len(output.detections) == 2
    assert output.orig_shape == (480, 640)
    assert output.inference_time_ms == 45.2


def test_detector_output_empty():
    """Test DetectorOutput with no detections."""
    output = DetectorOutput()
    assert len(output.detections) == 0
    assert output.orig_shape == (0, 0)
    assert output.inference_time_ms == 0.0


def test_base_detector_is_abstract():
    """Test that BaseDetector cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseDetector()
