"""Unit tests for BiomedCLIP classifier via BaseClassifier interface."""
import pytest
import numpy as np

from src.interfaces.classifier import ClassificationResult, BaseClassifier


def test_classification_result_dataclass():
    """Test ClassificationResult creation."""
    result = ClassificationResult(
        predicted_class="OSCC",
        class_scores={"OSCC": 0.7, "OPMD": 0.2, "Normal": 0.1},
        max_score=0.7,
        embedding=np.random.randn(512).astype(np.float32),
        inference_time_ms=120.5,
    )
    assert result.predicted_class == "OSCC"
    assert result.max_score == 0.7
    assert len(result.class_scores) == 3
    assert abs(sum(result.class_scores.values()) - 1.0) < 0.01
    assert result.embedding.shape == (512,)
    assert result.inference_time_ms == 120.5


def test_classification_result_defaults():
    """Test ClassificationResult defaults."""
    result = ClassificationResult()
    assert result.predicted_class == ""
    assert result.class_scores == {}
    assert result.max_score == 0.0
    assert result.embedding is None
    assert result.inference_time_ms == 0.0


def test_base_classifier_is_abstract():
    """Test that BaseClassifier cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseClassifier()
