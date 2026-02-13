"""Tests for FeedbackStore."""
import pytest

from src.utils.feedback import FeedbackStore


class TestFeedbackStore:
    """Tests for human-in-the-loop feedback collection."""

    def _make_store(self, tmp_path):
        return FeedbackStore(feedback_dir=str(tmp_path / "feedback"))

    def test_record_clinical_feedback(self, tmp_path):
        store = self._make_store(tmp_path)
        feedback_id = store.record_clinical_feedback(
            inference_id="inf_123",
            ai_prediction="OSCC",
            ai_risk_level="HIGH",
            ai_fusion_score=0.85,
            doctor_agrees=True,
            doctor_notes="Confirmed suspicious lesion",
        )
        assert feedback_id is not None
        assert len(feedback_id) == 8

        entries = store.load_all_feedback()
        assert len(entries) == 1
        assert entries[0]['ai_prediction'] == "OSCC"
        assert entries[0]['doctor_agrees'] is True

    def test_record_disagreement(self, tmp_path):
        store = self._make_store(tmp_path)
        store.record_clinical_feedback(
            inference_id="inf_456",
            ai_prediction="OSCC",
            ai_risk_level="HIGH",
            ai_fusion_score=0.9,
            doctor_agrees=False,
            doctor_classification="OPMD",
            doctor_notes="Looks more like leukoplakia",
        )
        entries = store.load_all_feedback()
        assert entries[0]['doctor_agrees'] is False
        assert entries[0]['doctor_classification'] == "OPMD"

    def test_record_ground_truth(self, tmp_path):
        store = self._make_store(tmp_path)
        store.record_ground_truth(
            inference_id="inf_789",
            ground_truth_class="OSCC",
            verification_method="biopsy",
            verified_by="Dr. Smith",
        )
        gts = store.load_ground_truths()
        assert len(gts) == 1
        assert gts[0]['ground_truth_class'] == "OSCC"
        assert gts[0]['verification_method'] == "biopsy"

    def test_agreement_rate(self, tmp_path):
        store = self._make_store(tmp_path)
        store.record_clinical_feedback(
            inference_id="1", ai_prediction="OSCC", ai_risk_level="HIGH",
            ai_fusion_score=0.8, doctor_agrees=True,
        )
        store.record_clinical_feedback(
            inference_id="2", ai_prediction="Normal", ai_risk_level="LOW",
            ai_fusion_score=0.3, doctor_agrees=True,
        )
        store.record_clinical_feedback(
            inference_id="3", ai_prediction="OSCC", ai_risk_level="HIGH",
            ai_fusion_score=0.9, doctor_agrees=False,
            doctor_classification="OPMD",
        )
        rate = store.compute_doctor_agreement_rate()
        assert rate['total'] == 3
        assert rate['agreed'] == 2
        assert rate['disagreed'] == 1
        assert rate['agreement_rate'] == pytest.approx(2 / 3)

    def test_empty_feedback(self, tmp_path):
        store = self._make_store(tmp_path)
        entries = store.load_all_feedback()
        assert entries == []
        rate = store.compute_doctor_agreement_rate()
        assert rate['total'] == 0
        assert rate['agreement_rate'] == 0.0
