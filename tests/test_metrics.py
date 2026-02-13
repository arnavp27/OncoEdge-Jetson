"""Tests for MetricsTracker."""
import pytest
import json
from pathlib import Path

import numpy as np

from src.utils.metrics import MetricsTracker
from src.config import MetricsConfig


class TestMetricsTracker:
    """Tests for metrics collection and computation."""

    def _make_tracker(self, tmp_path):
        config = MetricsConfig(
            enabled=True,
            log_file=str(tmp_path / "test_log.jsonl"),
        )
        return MetricsTracker(config=config)

    def test_log_inference(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        result = {
            'risk_level': 'HIGH',
            'risk_score': 9.5,
            'inference_id': 'test123',
            'detections': [
                {'fusion_score': 0.85, 'class': 'OSCC'},
                {'fusion_score': 0.60, 'class': 'OPMD'},
            ],
            'timing': {'pipeline_total_ms': 250.0},
            'model_info': {},
        }
        metadata = {'age': 55, 'tobacco_years': 15, 'lesion_duration': '> 6 weeks'}
        tracker.log_inference(result, metadata)

        log_path = Path(tmp_path / "test_log.jsonl")
        assert log_path.exists()
        with open(log_path) as f:
            entry = json.loads(f.readline())
        assert entry['risk_level'] == 'HIGH'
        assert entry['num_detections'] == 2

    def test_confidence_distribution(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        for score in [0.9, 0.8, 0.7, 0.6, 0.5]:
            result = {
                'detections': [{'fusion_score': score, 'class': 'OSCC'}],
                'timing': {},
                'model_info': {},
            }
            tracker.log_inference(result, {})

        dist = tracker.get_confidence_distribution()
        assert dist['count'] == 5
        assert dist['mean'] == pytest.approx(0.7)
        assert dist['min'] == pytest.approx(0.5)
        assert dist['max'] == pytest.approx(0.9)

    def test_confusion_matrix_with_ground_truth(self, tmp_path):
        tracker = self._make_tracker(tmp_path)

        tracker._class_predictions.append("OSCC")
        tracker._ground_truths.append("OSCC")
        tracker._class_predictions.append("Normal")
        tracker._ground_truths.append("OSCC")
        tracker._class_predictions.append("Normal")
        tracker._ground_truths.append("Normal")

        matrix = tracker.get_confusion_matrix()
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 1  # OSCC correctly predicted
        assert matrix[0, 2] == 1  # OSCC misclassified as Normal (FN)
        assert matrix[2, 2] == 1  # Normal correctly predicted

    def test_false_negative_rate(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker._class_predictions.extend(["OSCC", "OPMD", "Normal"])
        tracker._ground_truths.extend(["OSCC", "OPMD", "OSCC"])

        fnr = tracker.get_false_negative_rate()
        assert fnr == pytest.approx(1 / 3)

    def test_latency_stats(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker._latencies = [
            {'pipeline_total_ms': 100},
            {'pipeline_total_ms': 200},
            {'pipeline_total_ms': 300},
        ]
        stats = tracker.get_latency_stats()
        assert stats['mean_ms'] == pytest.approx(200.0)
        assert stats['p50_ms'] == pytest.approx(200.0)

    def test_reset(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker._confidence_scores = [0.5, 0.6]
        tracker._class_predictions = ["OSCC"]
        tracker.reset()
        assert len(tracker._confidence_scores) == 0
        assert len(tracker._class_predictions) == 0

    def test_disabled_tracker(self, tmp_path):
        config = MetricsConfig(enabled=False, log_file=str(tmp_path / "nolog.jsonl"))
        tracker = MetricsTracker(config=config)
        tracker.log_inference({'detections': [], 'timing': {}, 'model_info': {}}, {})
        assert not Path(tmp_path / "nolog.jsonl").exists()
