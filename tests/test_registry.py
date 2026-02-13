"""Tests for backend registry and format detection."""
import pytest

from src.backends.registry import (
    detect_format,
    _DETECTOR_BACKENDS,
    _CLASSIFIER_BACKENDS,
)
from src.backends.load_backends import load_all_backends


def test_detect_format_pt():
    assert detect_format("yolo11n-seg.pt") == ".pt"


def test_detect_format_onnx():
    assert detect_format("models/biomedclip/onnx/vision_encoder.onnx") == ".onnx"


def test_detect_format_engine():
    assert detect_format("models/biomedclip/trt/vision_encoder.engine") == ".engine"


def test_detect_format_bin():
    assert detect_format("models/biomedclip/open_clip_pytorch_model.bin") == ".bin"


def test_detect_format_hf_hub():
    assert detect_format("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224") == "hf-hub"


def test_load_all_backends_registers():
    """Test that loading backends registers at least core formats."""
    load_all_backends()
    assert ".pt" in _DETECTOR_BACKENDS, "Ultralytics .pt backend not registered"
    assert "hf-hub" in _CLASSIFIER_BACKENDS, "OpenCLIP hf-hub backend not registered"
    assert ".bin" in _CLASSIFIER_BACKENDS, "OpenCLIP .bin backend not registered"


def test_detector_registry_has_onnx():
    """Test that ONNX detector backend is registered if onnxruntime is available."""
    load_all_backends()
    try:
        import onnxruntime
        assert ".onnx" in _DETECTOR_BACKENDS
    except ImportError:
        pass  # OK if onnxruntime not installed


def test_classifier_registry_has_onnx():
    """Test that ONNX classifier backend is registered if onnxruntime is available."""
    load_all_backends()
    try:
        import onnxruntime
        assert ".onnx" in _CLASSIFIER_BACKENDS
    except ImportError:
        pass
