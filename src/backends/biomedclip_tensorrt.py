"""TensorRT backend for BiomedCLIP vision encoder (INT8 quantized).

Uses same precomputed text embeddings and logit scale as the ONNX backend.
Preprocessing via biomedclip_preprocess.py (single source of truth).

Only available on NVIDIA GPUs with TensorRT installed (Jetson).
"""
import json
import time

import numpy as np
from pathlib import Path

from src.interfaces.classifier import BaseClassifier, ClassificationResult
from src.backends.registry import register_classifier
from src.models.biomedclip_preprocess import preprocess_image


@register_classifier('.engine')
class TensorRTBiomedCLIPClassifier(BaseClassifier):
    """BiomedCLIP classifier using TensorRT INT8 engine.

    Identical contract to ONNXBiomedCLIPClassifier:
    - Uses precomputed text embeddings (numpy dot product)
    - Uses logit_scale from extraction script
    - Uses shared preprocessing from biomedclip_preprocess.py

    Only the inference execution differs (TensorRT vs ONNX Runtime).
    """

    def __init__(self, model_path: str, device: str = 'cuda',
                 text_embeddings_path: str = None,
                 logit_scale_path: str = None,
                 class_names: list = None, **kwargs):
        self.model_path = model_path
        self.device = device

        model_dir = Path(model_path).parent.parent
        self.text_embeddings_path = text_embeddings_path or str(model_dir / 'text_embeddings.npy')
        self.logit_scale_path = logit_scale_path or str(model_dir / 'logit_scale.json')
        self.class_names_list = class_names or ["OSCC", "OPMD", "Normal"]

        self.engine = None
        self.context = None
        self.text_features = None
        self.logit_scale = None

    def load(self) -> None:
        """Load TensorRT engine and precomputed artifacts."""
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Load precomputed text embeddings and logit scale
        self.text_features = np.load(self.text_embeddings_path)

        with open(self.logit_scale_path, 'r') as f:
            meta = json.load(f)
        self.logit_scale = meta['logit_scale']

        print(f"TensorRT BiomedCLIP loaded: {self.model_path}")
        print(f"  Text embeddings shape: {self.text_features.shape}")
        print(f"  Logit scale: {self.logit_scale:.2f}")

    def classify(self, image_patch: np.ndarray) -> ClassificationResult:
        """Classify a lesion patch using TensorRT engine.

        Same preprocessing and postprocessing as ONNX backend.
        Only the inference execution differs.

        Args:
            image_patch: numpy array (H, W, 3) uint8 RGB

        Returns:
            ClassificationResult with class probabilities
        """
        import pycuda.driver as cuda

        t0 = time.perf_counter()

        # Use shared preprocessing (single source of truth)
        input_tensor = preprocess_image(image_patch)  # (1, 3, 224, 224) float32

        # Allocate device memory and run TensorRT inference
        input_tensor = np.ascontiguousarray(input_tensor)
        d_input = cuda.mem_alloc(input_tensor.nbytes)
        output_shape = (1, 512)  # Vision encoder output
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        cuda.memcpy_htod(d_input, input_tensor)
        self.context.execute_v2([int(d_input), int(d_output)])
        cuda.memcpy_dtoh(output, d_output)

        image_features = output  # (1, 512)

        # L2 normalize
        image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)

        # Compute similarity with precomputed text features
        similarity = self.logit_scale * (image_features @ self.text_features.T)

        # Softmax
        exp_sim = np.exp(similarity - similarity.max(axis=-1, keepdims=True))
        scores = (exp_sim / exp_sim.sum(axis=-1, keepdims=True))[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        class_scores = {name: float(s) for name, s in zip(self.class_names_list, scores)}
        max_idx = int(scores.argmax())

        # Free device memory
        d_input.free()
        d_output.free()

        return ClassificationResult(
            predicted_class=self.class_names_list[max_idx],
            class_scores=class_scores,
            max_score=float(scores[max_idx]),
            embedding=image_features[0],
            inference_time_ms=elapsed_ms,
        )

    def get_class_names(self) -> list:
        return list(self.class_names_list)

    def get_model_info(self) -> dict:
        return {
            'format': '.engine (TensorRT INT8)',
            'model_path': self.model_path,
            'text_embeddings': self.text_embeddings_path,
            'logit_scale': self.logit_scale,
            'device': self.device,
        }
