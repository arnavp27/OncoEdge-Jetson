"""Human-in-the-loop feedback collection and evaluation.

Two modes:
1. Ground truth annotation: When pathology/biopsy results are available
2. Clinical feedback: Doctor's agreement/disagreement with AI prediction

Feedback is stored as JSONL for offline analysis and FNR tracking.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class FeedbackStore:
    """Persists clinical feedback alongside inference results.

    Storage format: JSONL (append-only, no schema migrations).
    """

    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback_log.jsonl"
        self.annotations_file = self.feedback_dir / "ground_truth_annotations.jsonl"

    def record_clinical_feedback(
        self,
        inference_id: str,
        ai_prediction: str,
        ai_risk_level: str,
        ai_fusion_score: float,
        doctor_agrees: bool,
        doctor_classification: Optional[str] = None,
        doctor_risk_override: Optional[str] = None,
        doctor_notes: str = "",
        patient_metadata: Optional[Dict] = None,
    ) -> str:
        """Record doctor's feedback on an AI prediction.

        Args:
            inference_id: Unique ID linking to the inference result
            ai_prediction: What the AI predicted (class name)
            ai_risk_level: AI's risk level assessment
            ai_fusion_score: AI's fusion confidence score
            doctor_agrees: Whether the doctor agrees with AI
            doctor_classification: Doctor's own classification if different
            doctor_risk_override: Doctor's risk level if different
            doctor_notes: Free-text clinical notes
            patient_metadata: Patient info dict

        Returns:
            feedback_id: Unique ID for this feedback entry
        """
        feedback_id = str(uuid.uuid4())[:8]
        entry = {
            'feedback_id': feedback_id,
            'inference_id': inference_id,
            'timestamp': datetime.now(tz=None).isoformat(),
            'ai_prediction': ai_prediction,
            'ai_risk_level': ai_risk_level,
            'ai_fusion_score': ai_fusion_score,
            'doctor_agrees': doctor_agrees,
            'doctor_classification': doctor_classification,
            'doctor_risk_override': doctor_risk_override,
            'doctor_notes': doctor_notes,
            'patient_metadata': patient_metadata,
        }

        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        return feedback_id

    def record_ground_truth(
        self,
        inference_id: str,
        ground_truth_class: str,
        verification_method: str,
        verified_by: str = "",
        notes: str = "",
    ) -> None:
        """Record verified ground truth for an inference.

        This is the gold standard for computing FNR and confusion matrices.

        Args:
            inference_id: Unique ID linking to the inference result
            ground_truth_class: Verified class ("OSCC", "OPMD", "Normal")
            verification_method: How it was verified ("biopsy", "clinical_followup", "specialist_review")
            verified_by: Name/ID of verifier
            notes: Additional notes
        """
        entry = {
            'inference_id': inference_id,
            'timestamp': datetime.now(tz=None).isoformat(),
            'ground_truth_class': ground_truth_class,
            'verification_method': verification_method,
            'verified_by': verified_by,
            'notes': notes,
        }

        with open(self.annotations_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def load_all_feedback(self) -> List[dict]:
        """Load all feedback entries for analysis."""
        entries = []
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def load_ground_truths(self) -> List[dict]:
        """Load all ground truth annotations."""
        entries = []
        if self.annotations_file.exists():
            with open(self.annotations_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def compute_doctor_agreement_rate(self) -> dict:
        """Compute agreement rate between AI and doctors.

        Returns:
            dict with total, agreed, disagreed, agreement_rate
        """
        feedback = self.load_all_feedback()
        if not feedback:
            return {'total': 0, 'agreed': 0, 'disagreed': 0, 'agreement_rate': 0.0}

        agreed = sum(1 for f in feedback if f.get('doctor_agrees'))
        return {
            'total': len(feedback),
            'agreed': agreed,
            'disagreed': len(feedback) - agreed,
            'agreement_rate': agreed / len(feedback),
        }
