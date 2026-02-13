"""Performance and clinical metrics tracking.

Tracks:
- Inference latency (per-component and total)
- Confidence score distributions
- Confusion matrix (when ground truth available)
- False negative monitoring (critical for medical use)
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class MetricsTracker:
    """Collects and persists inference metrics.

    Logs to JSONL for append-only, no-schema-migration persistence.
    In-memory accumulators provide real-time statistics.
    """

    def __init__(self, config=None):
        """Initialize metrics tracker.

        Args:
            config: MetricsConfig dataclass, or None for defaults.
        """
        self.enabled = config.enabled if config else True
        self.log_file = Path(config.log_file) if config else Path("data/metrics/inference_log.jsonl")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # In-memory accumulators (reset per session)
        self._latencies: List[Dict[str, float]] = []
        self._confidence_scores: List[float] = []
        self._class_predictions: List[str] = []
        self._ground_truths: List[Optional[str]] = []

    def log_inference(self, result: dict, patient_metadata: dict,
                      ground_truth: Optional[str] = None) -> None:
        """Log a single inference result.

        Args:
            result: Pipeline output dict
            patient_metadata: Patient info dict
            ground_truth: Optional ground truth class label for evaluation
        """
        if not self.enabled:
            return

        entry = {
            'timestamp': datetime.now(tz=None).isoformat(),
            'inference_id': result.get('inference_id', ''),
            'risk_level': result.get('risk_level'),
            'risk_score': result.get('risk_score'),
            'num_detections': len(result.get('detections', [])),
            'timing': result.get('timing', {}),
            'model_info': {
                k: {mk: mv for mk, mv in v.items() if isinstance(mv, (str, int, float, bool))}
                for k, v in result.get('model_info', {}).items()
                if isinstance(v, dict)
            },
            'patient_age': patient_metadata.get('age'),
            'tobacco_years': patient_metadata.get('tobacco_years'),
            'lesion_duration': patient_metadata.get('lesion_duration'),
            'ground_truth': ground_truth,
        }

        # Per-detection metrics
        for det in result.get('detections', []):
            self._confidence_scores.append(det['fusion_score'])
            self._class_predictions.append(det['class'])
            self._ground_truths.append(ground_truth)

        if result.get('timing'):
            self._latencies.append(result['timing'])

        # Append to JSONL log
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except OSError:
            pass  # Silently fail on write errors to not block inference

    def get_confusion_matrix(self, class_names: List[str] = None) -> np.ndarray:
        """Compute confusion matrix from accumulated predictions with ground truth.

        Only includes samples where ground_truth is not None.

        Args:
            class_names: Ordered list of class names. Defaults to ["OSCC", "OPMD", "Normal"].

        Returns:
            np.ndarray: (C, C) confusion matrix, rows=true, cols=predicted
        """
        if class_names is None:
            class_names = ["OSCC", "OPMD", "Normal"]

        n = len(class_names)
        matrix = np.zeros((n, n), dtype=int)
        name_to_idx = {name: i for i, name in enumerate(class_names)}

        for pred, gt in zip(self._class_predictions, self._ground_truths):
            if gt is not None and gt in name_to_idx and pred in name_to_idx:
                matrix[name_to_idx[gt], name_to_idx[pred]] += 1

        return matrix

    def get_false_negative_rate(self, positive_classes: List[str] = None) -> float:
        """Calculate false negative rate for critical classes.

        In oral cancer screening, missing OSCC or OPMD is the critical failure mode.
        FNR = FN / (FN + TP) for the positive classes.

        Args:
            positive_classes: Classes considered "positive" (default: ["OSCC", "OPMD"])

        Returns:
            float: False negative rate (0.0 - 1.0)
        """
        if positive_classes is None:
            positive_classes = ["OSCC", "OPMD"]

        fn = 0
        tp = 0

        for pred, gt in zip(self._class_predictions, self._ground_truths):
            if gt is None:
                continue
            if gt in positive_classes:
                if pred in positive_classes:
                    tp += 1
                else:
                    fn += 1

        if tp + fn == 0:
            return 0.0
        return fn / (fn + tp)

    def get_confidence_distribution(self) -> dict:
        """Return confidence score statistics.

        Returns:
            dict with mean, std, min, max, median, p5, p95, count
        """
        if not self._confidence_scores:
            return {}
        scores = np.array(self._confidence_scores)
        return {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'median': float(np.median(scores)),
            'p5': float(np.percentile(scores, 5)),
            'p95': float(np.percentile(scores, 95)),
            'count': len(scores),
        }

    def get_latency_stats(self) -> dict:
        """Return latency statistics for pipeline total time.

        Returns:
            dict with mean_ms, p50_ms, p95_ms, max_ms
        """
        if not self._latencies:
            return {}
        totals = [l.get('pipeline_total_ms', 0) for l in self._latencies if 'pipeline_total_ms' in l]
        if not totals:
            return {}
        arr = np.array(totals)
        return {
            'mean_ms': float(arr.mean()),
            'p50_ms': float(np.median(arr)),
            'p95_ms': float(np.percentile(arr, 95)),
            'max_ms': float(arr.max()),
        }

    def reset(self):
        """Reset in-memory accumulators."""
        self._latencies.clear()
        self._confidence_scores.clear()
        self._class_predictions.clear()
        self._ground_truths.clear()
