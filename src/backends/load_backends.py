"""Import all backend modules to trigger @register_detector/@register_classifier decorators."""


def load_all_backends():
    """Import all backend modules. Call once at startup.

    Core backends (ultralytics, open_clip) are always loaded.
    Optional backends (ONNX, TensorRT) are loaded only if their
    dependencies are available.
    """
    # Core backends -- always available
    from src.backends import yolo_ultralytics      # registers .pt
    from src.backends import biomedclip_openclip   # registers hf-hub, .bin

    # Optional backends -- import only if dependencies available
    try:
        from src.backends import yolo_onnx         # registers .onnx for detector
    except ImportError:
        pass

    try:
        from src.backends import biomedclip_onnx   # registers .onnx for classifier
    except ImportError:
        pass

    try:
        from src.backends import biomedclip_tensorrt  # registers .engine
    except ImportError:
        pass
