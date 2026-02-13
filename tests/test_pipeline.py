"""Integration tests for pipeline components."""
import pytest
import numpy as np

from src.config import load_config, OncoEdgeConfig, ClinicalConfig, FusionConfig
from src.pipeline.fusion import FusionModule
from src.pipeline.decision_tree import ClinicalDecisionTree


class TestFusionModule:
    """Tests for FusionModule with configurable methods."""

    def test_gate_fusion_above_threshold(self):
        fusion = FusionModule(method="gate", gate_threshold=0.5)
        # YOLO 0.7 >= 0.5 → approved, returns CLIP score directly
        assert fusion.combine(0.7, 0.85) == pytest.approx(0.85)

    def test_gate_fusion_below_threshold(self):
        fusion = FusionModule(method="gate", gate_threshold=0.5)
        # YOLO 0.3 < 0.5 → rejected
        assert fusion.combine(0.3, 0.85) == -1.0

    def test_gate_fusion_at_threshold(self):
        fusion = FusionModule(method="gate", gate_threshold=0.5)
        # YOLO 0.5 >= 0.5 → approved (boundary)
        assert fusion.combine(0.5, 0.75) == pytest.approx(0.75)

    def test_gate_clip_score_passes_through(self):
        """Verify YOLO conf does NOT affect the output score in gate mode."""
        fusion = FusionModule(method="gate", gate_threshold=0.5)
        # High YOLO and low YOLO both give the same CLIP score
        assert fusion.combine(0.9, 0.60) == fusion.combine(0.5, 0.60)

    def test_multiplicative_fusion(self):
        fusion = FusionModule(method="multiplicative")
        assert fusion.combine(0.8, 0.9) == pytest.approx(0.72)

    def test_weighted_fusion(self):
        fusion = FusionModule(method="weighted")
        result = fusion.combine(0.8, 0.9)
        expected = 0.4 * 0.8 + 0.6 * 0.9
        assert result == pytest.approx(expected)

    def test_combine_batch_gate(self):
        fusion = FusionModule(method="gate", gate_threshold=0.5)
        results = fusion.combine_batch([0.7, 0.3, 0.6], [0.8, 0.9, 0.5])
        assert results[0] == pytest.approx(0.8)  # approved
        assert results[1] == -1.0                  # rejected
        assert results[2] == pytest.approx(0.5)   # approved


class TestClinicalDecisionTree:
    """Tests for ClinicalDecisionTree with new risk formula.

    Formula: risk_score = clip_score * 10 + alpha * tobacco_norm + beta * smoking_norm
    Defaults: alpha=2.0, beta=1.5, tobacco_significant=15, smoking_significant=15
    """

    def test_high_risk_clip_alone(self):
        """High CLIP score (0.85) alone → 8.5 > 8.0 → HIGH."""
        tree = ClinicalDecisionTree()
        result = tree.assess_risk(vision_score=0.85, tobacco_years=0, smoking_years=0)
        assert result['level'] == 'HIGH'
        assert result['score'] == pytest.approx(8.5)

    def test_high_risk_with_tobacco_and_smoking(self):
        """Moderate CLIP + heavy tobacco + heavy smoking → HIGH."""
        tree = ClinicalDecisionTree()
        # 0.6*10 + 2.0*(15/15) + 1.5*(15/15) = 6.0 + 2.0 + 1.5 = 9.5
        result = tree.assess_risk(vision_score=0.6, tobacco_years=15, smoking_years=15)
        assert result['level'] == 'HIGH'
        assert result['score'] == pytest.approx(9.5)

    def test_medium_risk(self):
        """Moderate CLIP + some tobacco → MEDIUM."""
        tree = ClinicalDecisionTree()
        # 0.5*10 + 2.0*(7/15) + 1.5*0 = 5.0 + 0.933 = 5.933
        result = tree.assess_risk(vision_score=0.5, tobacco_years=7, smoking_years=0)
        assert result['level'] == 'MEDIUM'
        assert 5.0 < result['score'] <= 8.0

    def test_low_risk(self):
        tree = ClinicalDecisionTree()
        result = tree.assess_risk(vision_score=0.2, tobacco_years=0, smoking_years=0)
        assert result['level'] == 'LOW'
        assert result['score'] == pytest.approx(2.0)

    def test_tobacco_normalization_capped_at_1(self):
        """Tobacco years beyond threshold should cap at 1.0."""
        tree = ClinicalDecisionTree()
        # 30 years / 15 threshold = 2.0, capped to 1.0
        result = tree.assess_risk(vision_score=0.0, tobacco_years=30, smoking_years=0)
        assert result['score'] == pytest.approx(2.0)  # 0 + 2.0*1.0 + 0
        assert result['breakdown']['tobacco_norm'] == 1.0

    def test_smoking_normalization(self):
        tree = ClinicalDecisionTree()
        # 10 years / 15 threshold = 0.667
        result = tree.assess_risk(vision_score=0.0, tobacco_years=0, smoking_years=10)
        assert result['breakdown']['smoking_norm'] == pytest.approx(10 / 15, abs=0.01)
        assert result['score'] == pytest.approx(1.5 * (10 / 15), abs=0.02)

    def test_breakdown_included(self):
        tree = ClinicalDecisionTree()
        result = tree.assess_risk(vision_score=0.7, tobacco_years=10, smoking_years=5)
        bd = result['breakdown']
        assert bd['clip_base'] == pytest.approx(7.0)
        assert bd['tobacco_norm'] == pytest.approx(10 / 15, abs=0.01)
        assert bd['smoking_norm'] == pytest.approx(5 / 15, abs=0.01)

    def test_config_driven(self):
        config = ClinicalConfig(
            high_risk_threshold=10.0,
            medium_risk_threshold=6.0,
            alpha=3.0,
            tobacco_significant_years=10,
            beta=2.0,
            smoking_significant_years=20,
        )
        tree = ClinicalDecisionTree(config=config)
        # 0.5*10 + 3.0*(10/10) + 2.0*(10/20) = 5.0 + 3.0 + 1.0 = 9.0
        result = tree.assess_risk(vision_score=0.5, tobacco_years=10, smoking_years=10)
        assert result['score'] == pytest.approx(9.0)
        assert result['level'] == 'MEDIUM'  # 9.0 < 10.0 threshold

    def test_backward_compatible_kwargs(self):
        """Old-style calls with age/lesion_duration should not crash."""
        tree = ClinicalDecisionTree()
        result = tree.assess_risk(
            vision_score=0.5, tobacco_years=5, smoking_years=0,
            age=55, lesion_duration="> 6 weeks"
        )
        # Extra kwargs are ignored
        assert result['level'] in ['HIGH', 'MEDIUM', 'LOW']

    def test_get_risk_factors_both(self):
        tree = ClinicalDecisionTree()
        factors = tree.get_risk_factors(tobacco_years=15, smoking_years=10)
        assert len(factors) == 2

    def test_get_risk_factors_none(self):
        tree = ClinicalDecisionTree()
        factors = tree.get_risk_factors(tobacco_years=0, smoking_years=0)
        assert len(factors) == 0


class TestConfig:
    """Tests for configuration loading."""

    def test_default_config(self):
        config = OncoEdgeConfig()
        assert config.detector.model_path == "yolo11n-seg.pt"
        assert config.detector.conf_threshold == 0.25
        assert config.clinical.high_risk_threshold == 8.0
        assert config.clinical.alpha == 2.0
        assert config.clinical.beta == 1.5

    def test_load_config_from_yaml(self):
        config = load_config("config/default.yaml")
        assert config.detector.model_path == "yolo11n-seg.pt"
        assert config.device.preferred == "auto"
        assert config.fusion.method == "gate"
        assert config.fusion.gate_threshold == 0.5

    def test_device_resolution(self):
        config = OncoEdgeConfig()
        device = config.device.resolved
        assert device in ['cuda', 'mps', 'cpu']
