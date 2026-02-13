"""ONNX Runtime backend for BiomedCLIP vision encoder.

Uses:
- ONNX vision encoder from export_vision_encoder_onnx.py
- Precomputed text embeddings from precompute_text_embeddings.py
- Logit scale from extract_clip_logit_scale.py
- Preprocessing from biomedclip_preprocess.py (single source of truth)

This eliminates the text encoder entirely at inference time.
Only the vision encoder runs through ONNX Runtime.
"""
import json
import time

import numpy as np
import onnxruntime as ort
from pathlib import Path

from src.interfaces.classifier import BaseClassifier, ClassificationResult
from src.backends.registry import register_classifier
from src.models.biomedclip_preprocess import preprocess_image


@register_classifier('.onnx')
class ONNXBiomedCLIPClassifier(BaseClassifier):
    """BiomedCLIP classifier using ONNX vision encoder + precomputed text embeddings.

    Only the vision encoder runs through ONNX Runtime.
    Similarity is computed as a numpy dot product + softmax.
    No PyTorch required at inference time.
    """

    def __init__(self, model_path: str, device: str = 'cpu',
                 text_embeddings_path: str = None,
                 logit_scale_path: str = None,
                 class_names: list = None, **kwargs):
        self.model_path = model_path
        self.device = device

        # Derive paths from model_path directory if not specified
        model_dir = Path(model_path).parent.parent  # e.g., models/biomedclip/
        self.text_embeddings_path = text_embeddings_path or str(model_dir / 'text_embeddings.npy')
        self.logit_scale_path = logit_scale_path or str(model_dir / 'logit_scale.json')
        self.class_names_list = class_names or ["OSCC", "OPMD", "Normal"]

        self.session = None
        self.text_features = None
        self.logit_scale = None

    def load(self) -> None:
        """Load ONNX session and precomputed artifacts."""
        # Select execution providers based on device
        providers = ['CPUExecutionProvider']
        if self.device in ('cuda', '0'):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.session = ort.InferenceSession(self.model_path, providers=providers)

        # Load precomputed text embeddings
        self.text_features = np.load(self.text_embeddings_path)  # (3, 512)

        # Load logit scale
        with open(self.logit_scale_path, 'r') as f:
            meta = json.load(f)
        self.logit_scale = meta['logit_scale']

        print(f"ONNX BiomedCLIP loaded: {self.model_path}")
        print(f"  Text embeddings: {self.text_embeddings_path} shape={self.text_features.shape}")
        print(f"  Logit scale: {self.logit_scale:.2f}")

    def classify(self, image_patch: np.ndarray) -> ClassificationResult:
        """Classify a lesion patch using ONNX vision encoder.

        Uses biomedclip_preprocess.preprocess_image() for exact preprocessing
        parity with calibration and full-model inference.

        Args:
            image_patch: numpy array (H, W, 3) uint8 RGB

        Returns:
            ClassificationResult with class probabilities
        """
        t0 = time.perf_counter()

        # Use shared preprocessing (single source of truth)
        input_tensor = preprocess_image(image_patch)  # (1, 3, 224, 224) float32

        # Run vision encoder
        input_name = self.session.get_inputs()[0].name
        image_features = self.session.run(None, {input_name: input_tensor})[0]  # (1, D)

        # L2 normalize
        image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)

        # Compute similarity with precomputed text features
        similarity = self.logit_scale * (image_features @ self.text_features.T)  # (1, 3)

        # Softmax
        exp_sim = np.exp(similarity - similarity.max(axis=-1, keepdims=True))
        scores = (exp_sim / exp_sim.sum(axis=-1, keepdims=True))[0]  # (3,)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        class_scores = {name: float(s) for name, s in zip(self.class_names_list, scores)}
        max_idx = int(scores.argmax())

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
            'format': '.onnx (vision encoder only)',
            'model_path': self.model_path,
            'text_embeddings': self.text_embeddings_path,
            'logit_scale': self.logit_scale,
            'device': self.device,
        }
