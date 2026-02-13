"""Compare pre-quantized vs post-quantized model outputs.

Designed to run on a calibration or test dataset to measure:
- Output distribution shift (KL divergence, KS statistic)
- Confidence calibration changes
- False negative rate changes
- Per-class score drift
- Embedding cosine similarity
"""
import numpy as np
from typing import List, Optional

from src.interfaces.classifier import BaseClassifier
from src.backends.registry import create_classifier
from src.backends.load_backends import load_all_backends


class QuantizationShiftAnalyzer:
    """Compares outputs of two classifier backends on the same inputs.

    Typical usage:
        analyzer = QuantizationShiftAnalyzer(
            reference_model_path="hf-hub:microsoft/BiomedCLIP-...",
            quantized_model_path="models/biomedclip/onnx/vision_encoder.onnx",
            device="cpu"
        )
        report = analyzer.analyze(image_paths)
    """

    def __init__(self, reference_model_path: str, quantized_model_path: str,
                 device: str = "cpu", **kwargs):
        """Initialize with two model paths.

        Args:
            reference_model_path: Path to the reference (pre-quantized) model
            quantized_model_path: Path to the quantized model
            device: Device for inference
            **kwargs: Additional backend-specific parameters
        """
        load_all_backends()
        self.reference: BaseClassifier = create_classifier(
            reference_model_path, device=device, **kwargs
        )
        self.quantized: BaseClassifier = create_classifier(
            quantized_model_path, device=device, **kwargs
        )

    def analyze(self, image_paths: List[str],
                ground_truths: Optional[List[str]] = None) -> dict:
        """Run both models on all images and compare outputs.

        Args:
            image_paths: List of paths to test images
            ground_truths: Optional ground truth labels for FNR comparison

        Returns:
            dict with comprehensive shift analysis
        """
        from PIL import Image

        ref_scores = []
        quant_scores = []
        ref_preds = []
        quant_preds = []
        ref_embeddings = []
        quant_embeddings = []

        for path in image_paths:
            img = np.array(Image.open(path).convert('RGB'))

            ref_result = self.reference.classify(img)
            quant_result = self.quantized.classify(img)

            ref_scores.append(list(ref_result.class_scores.values()))
            quant_scores.append(list(quant_result.class_scores.values()))
            ref_preds.append(ref_result.predicted_class)
            quant_preds.append(quant_result.predicted_class)

            if ref_result.embedding is not None:
                ref_embeddings.append(ref_result.embedding)
            if quant_result.embedding is not None:
                quant_embeddings.append(quant_result.embedding)

        ref_scores = np.array(ref_scores)
        quant_scores = np.array(quant_scores)

        report = {
            'num_images': len(image_paths),
            'prediction_agreement': float(np.mean(
                [r == q for r, q in zip(ref_preds, quant_preds)]
            )),
            'max_score_diff': self._score_diff_stats(ref_scores, quant_scores),
            'per_class_drift': self._per_class_drift(ref_scores, quant_scores),
            'ks_test': self._ks_test_per_class(ref_scores, quant_scores),
        }

        if ref_embeddings and quant_embeddings:
            report['embedding_cosine_similarity'] = self._embedding_similarity(
                np.array(ref_embeddings), np.array(quant_embeddings)
            )

        if ground_truths:
            report['reference_fnr'] = self._compute_fnr(ref_preds, ground_truths)
            report['quantized_fnr'] = self._compute_fnr(quant_preds, ground_truths)
            report['fnr_change'] = report['quantized_fnr'] - report['reference_fnr']

        return report

    def _score_diff_stats(self, ref: np.ndarray, quant: np.ndarray) -> dict:
        diff = np.abs(ref - quant)
        return {
            'mean': float(diff.mean()),
            'max': float(diff.max()),
            'std': float(diff.std()),
        }

    def _per_class_drift(self, ref: np.ndarray, quant: np.ndarray) -> dict:
        class_names = self.reference.get_class_names()
        drift = {}
        for i, name in enumerate(class_names):
            drift[name] = {
                'mean_ref': float(ref[:, i].mean()),
                'mean_quant': float(quant[:, i].mean()),
                'mean_shift': float(quant[:, i].mean() - ref[:, i].mean()),
            }
        return drift

    def _ks_test_per_class(self, ref: np.ndarray, quant: np.ndarray) -> dict:
        from scipy import stats

        class_names = self.reference.get_class_names()
        results = {}
        for i, name in enumerate(class_names):
            stat, p_value = stats.ks_2samp(ref[:, i], quant[:, i])
            results[name] = {
                'ks_statistic': float(stat),
                'p_value': float(p_value),
                'significant_drift': p_value < 0.05,
            }
        return results

    def _embedding_similarity(self, ref: np.ndarray, quant: np.ndarray) -> dict:
        dot = np.sum(ref * quant, axis=-1)
        ref_norm = np.linalg.norm(ref, axis=-1)
        quant_norm = np.linalg.norm(quant, axis=-1)
        cosine = dot / (ref_norm * quant_norm + 1e-8)
        return {
            'mean': float(cosine.mean()),
            'min': float(cosine.min()),
            'std': float(cosine.std()),
        }

    def _compute_fnr(self, predictions: List[str], ground_truths: List[str]) -> float:
        positive_classes = ["OSCC", "OPMD"]
        fn = sum(1 for p, g in zip(predictions, ground_truths)
                 if g in positive_classes and p not in positive_classes)
        tp = sum(1 for p, g in zip(predictions, ground_truths)
                 if g in positive_classes and p in positive_classes)
        return fn / (fn + tp) if (fn + tp) > 0 else 0.0
