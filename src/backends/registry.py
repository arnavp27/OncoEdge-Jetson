"""Backend registry for model format auto-detection and loading."""
from pathlib import Path
from typing import Dict, Type

from src.interfaces.detector import BaseDetector
from src.interfaces.classifier import BaseClassifier

# Registry maps: file extension -> backend class
_DETECTOR_BACKENDS: Dict[str, Type[BaseDetector]] = {}
_CLASSIFIER_BACKENDS: Dict[str, Type[BaseClassifier]] = {}


def register_detector(format_ext: str):
    """Decorator to register a detector backend for a file extension.

    Usage:
        @register_detector('.pt')
        class UltralyticsDetector(BaseDetector): ...
    """
    def decorator(cls):
        _DETECTOR_BACKENDS[format_ext] = cls
        return cls
    return decorator


def register_classifier(format_ext: str):
    """Decorator to register a classifier backend for a file extension."""
    def decorator(cls):
        _CLASSIFIER_BACKENDS[format_ext] = cls
        return cls
    return decorator


def detect_format(model_path: str) -> str:
    """Detect model format from file extension or hub prefix.

    Args:
        model_path: Path to model file or HuggingFace hub string

    Returns:
        Extension string: '.pt', '.onnx', '.engine', '.bin', 'hf-hub'
    """
    if model_path.startswith('hf-hub:'):
        return 'hf-hub'
    return Path(model_path).suffix.lower()


def create_detector(model_path: str, device: str, **kwargs) -> BaseDetector:
    """Factory: create detector from model path using registry.

    Auto-detects format from file extension and instantiates
    the appropriate backend.

    Args:
        model_path: Path to model file
        device: Device string ('cuda', 'mps', 'cpu', '0', etc.)
        **kwargs: Additional backend-specific parameters

    Returns:
        Loaded BaseDetector instance
    """
    fmt = detect_format(model_path)
    if fmt not in _DETECTOR_BACKENDS:
        available = list(_DETECTOR_BACKENDS.keys())
        raise ValueError(
            f"No detector backend registered for format '{fmt}'. "
            f"Available: {available}"
        )
    backend_cls = _DETECTOR_BACKENDS[fmt]
    instance = backend_cls(model_path=model_path, device=device, **kwargs)
    instance.load()
    return instance


def create_classifier(model_path: str, device: str, **kwargs) -> BaseClassifier:
    """Factory: create classifier from model path using registry.

    Args:
        model_path: Path to model file or hub string
        device: Device string
        **kwargs: Additional backend-specific parameters

    Returns:
        Loaded BaseClassifier instance
    """
    fmt = detect_format(model_path)
    if fmt not in _CLASSIFIER_BACKENDS:
        available = list(_CLASSIFIER_BACKENDS.keys())
        raise ValueError(
            f"No classifier backend registered for format '{fmt}'. "
            f"Available: {available}"
        )
    backend_cls = _CLASSIFIER_BACKENDS[fmt]
    instance = backend_cls(model_path=model_path, device=device, **kwargs)
    instance.load()
    return instance


def get_registered_detector_formats():
    """Return list of registered detector format extensions."""
    return list(_DETECTOR_BACKENDS.keys())


def get_registered_classifier_formats():
    """Return list of registered classifier format extensions."""
    return list(_CLASSIFIER_BACKENDS.keys())
